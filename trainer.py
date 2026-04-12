
"""
Training module for Switti models with textual inversion and disentanglement capabilities.

This module provides training classes for Switti models, supporting both standard
textual inversion training and content-style disentanglement training with
multi-scale loss weighting and orthogonal loss constraints.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

import dist
from models import VQVAE, Switti
from models.pipeline import SwittiPipeline
from utils.amp_sc import AmpOptimizer
from utils.misc import TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

EVAL_PROMPTS = {
    "object": [
        "a {}",
        "a {} in the jungle",
        "a {} in the snow",
        "a {} on the beach",
        "a {} with a tree and autumn leaves in the background",
        "a {} on top of pink fabric",
        "a {} made of gold",
        "a {} made of glass",
    ],
    "style": [
        "a {}",
        "a dog in style of {}",
        "a dog in {}",
        "a cat in style of {}",
        "a cat in {}",
        "a bunny in style of {}",
        "a bunny in {}",
        "a man in style of {}",
        "a man in {}",
    ],
}

EVAL_PROMPTS_DISENTANGLEMENT = {
    "content": [
        "a {}",
        "a cat {}",
        "a white cat {}",
        "a white cat {} on the beach",
    ],
    "style": [
        "a {}",
        "a dog in fantastic {} style",
        "a cat in fantastic {} style",
        "a fish in fantastic {} style",
    ],
    "content_style": [
        "A white cat {} in fantastic {} style",
        "A white cat {} is smiling in fantastic {} style ",
    ],
}


class SwittiTrainer:
    """
    Base trainer class for Switti models with textual inversion.

    This class handles the training loop for Switti models, including
    multi-scale loss computation, gradient accumulation, and tensorboard logging.
    """

    def __init__(
        self,
        dataloader,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        pipe: SwittiPipeline,
        vae_local: VQVAE,
        switti_wo_ddp: Switti,
        switti: DDP,
        optimizer: AmpOptimizer,
        label_smooth: float,
        placeholder_token_ids: List[int] = [],
        placeholder_token_ids_2: List[int] = [],
        args=None,
        **kwargs,
    ):
        """
        Initialize the SwittiTrainer.

        Args:
            dataloader: Data loader for training data
            device: PyTorch device for computation
            patch_nums: Tuple of patch numbers for each scale
            resos: Tuple of resolutions for each scale
            pipe: Switti pipeline for text encoding and generation
            vae_local: Local VAE model for image encoding/decoding
            switti_wo_ddp: Switti model without distributed wrapper
            switti: Distributed Switti model
            optimizer: Optimizer with mixed precision support
            label_smooth: Label smoothing factor for loss computation
            placeholder_token_ids: List of placeholder token IDs for text encoder 1
            placeholder_token_ids_2: List of placeholder token IDs for text encoder 2
            args: Training arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.dataloader = iter(dataloader)
        self.args = args

        self.switti, self.vae_local, self.quantize_local = (
            switti,
            vae_local,
            vae_local.quantize,
        )
        self.switti_wo_ddp: Switti = switti_wo_ddp  # after torch.compile
        self.optimizer = optimizer
        self.pipe = pipe
        self.switti_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction="none")
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.specific_scale = args.train_specific_scale
        if self.specific_scale:
            self.loss_weight = torch.zeros_like(self.loss_weight)
            scales = args.train_specific_scale.split("_")
            scales = [int(scale) for scale in scales]

            normalize_factor = sum(
                patch_nums[int(scale)] * patch_nums[int(scale)] for scale in scales
            )

            print("Scales", scales)
            print("Normalize factor", normalize_factor)

            cur_idx = 0
            for i_pn in range(len(patch_nums)):
                if i_pn in scales:
                    self.loss_weight[:, cur_idx : cur_idx + patch_nums[i_pn] * patch_nums[i_pn]] = (
                        1 / normalize_factor
                    )
                cur_idx += patch_nums[i_pn] * patch_nums[i_pn]

            print("Loss weight", self.loss_weight)

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for pn in patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn
        self.device = device
        self.grad_accum = args.grad_accum
        self.embed_noise_std = args.embed_noise_std

        self.placeholder_token_ids = placeholder_token_ids
        self.placeholder_token_ids_2 = placeholder_token_ids_2

        self.eval_prompts = EVAL_PROMPTS[args.learnable_property]

        for i, prompt in enumerate(self.eval_prompts):
            self.eval_prompts[i] = prompt.format(args.placeholder_token_use)

    def train_step(
        self,
        g_it: int,
        tb_lg: TensorboardLogger,
        orig_embeds_params: torch.Tensor,
        orig_embeds_params_2: torch.Tensor,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """
        Perform a single training step.

        Args:
            g_it: Global iteration number
            tb_lg: Tensorboard logger for metrics
            orig_embeds_params: Original embedding parameters for text encoder 1
            orig_embeds_params_2: Original embedding parameters for text encoder 2

        Returns:
            Tuple of (gradient_norm, scale_log2)
        """
        # forward
        self.pipe.text_encoder.transformer.train()
        self.pipe.text_encoder_2.transformer.train()

        for accum_iter in range(self.grad_accum):
            image, prompt = next(self.dataloader)

            inp_B3HW = image.to(self.device, non_blocking=True)
            inp_B3HW = F.interpolate(
                inp_B3HW,
                size=(self.resos[-1], self.resos[-1]),
                mode="bicubic",
            )

            B, V = inp_B3HW.size(0), self.vae_local.vocab_size

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(
                inp_B3HW, noise_std=self.embed_noise_std
            )
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_switti_input(gt_idx_Bl)
            if self.args.uncond_proba > 0:
                cond_uncond_choice = torch.bernoulli(torch.full((B,), self.args.uncond_proba))
                for i_, p_ in enumerate(cond_uncond_choice):
                    if p_ == 1:
                        prompt[i_] = ""
            (
                prompt_embeds,
                pooled_prompt_embeds,
                prompt_attn_bias,
            ) = self.pipe.encode_prompt(prompt, encode_null=False)

            with self.optimizer.amp_ctx:
                batch_embed = prompt_embeds.shape[0]

                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    batch_height=batch_embed * [self.resos[-1]],
                    batch_width=batch_embed * [self.resos[-1]],
                )

                loss = self.train_loss(
                    logits_BLV.view(-1, V),
                    gt_BL.view(-1),
                ).view(B, -1)
                loss = loss.mul(self.loss_weight).sum(dim=-1).mean()

            # backward
            is_stepping = (accum_iter + 1) == self.grad_accum
            grad_norm, scale_log2 = self.optimizer.backward_clip_step(
                loss=loss,
                is_stepping=is_stepping,
            )

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones(
                (len(self.pipe.text_encoder.tokenizer),), dtype=torch.bool
            )
            index_no_updates[
                min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1
            ] = False
            index_no_updates_2 = torch.ones(
                (len(self.pipe.text_encoder_2.tokenizer),), dtype=torch.bool
            )
            index_no_updates_2[
                min(self.placeholder_token_ids_2) : max(self.placeholder_token_ids_2) + 1
            ] = False

            with torch.no_grad():
                self.pipe.text_encoder.transformer.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]
                self.pipe.text_encoder_2.transformer.get_input_embeddings().weight[
                    index_no_updates_2
                ] = orig_embeds_params_2[index_no_updates_2]

        # log to tensorboard
        if g_it > 0 and g_it % self.args.log_iters == 0:
            # recalculate logits in .eval() mode to log acc
            self.switti.eval()
            if self.args.use_gradient_checkpointing:
                self.switti.disable_gradient_checkpointing()
            with torch.no_grad(), self.optimizer.amp_ctx:
                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    batch_height=batch_embed * [self.resos[-1]],
                    batch_width=batch_embed * [self.resos[-1]],
                )

            # Compute cluster usage
            pred_BL = logits_BLV.data.argmax(dim=-1)
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float().cuda()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100

            logits_lg = dict()
            kw = dict(z_voc_usage=cluster_usage, acc_total=0.0, L_total=0.0)
            for si, (bg, ed) in enumerate(self.begin_ends):
                pred = logits_BLV.data[:, bg:ed].reshape(-1, V)
                tar = gt_BL[:, bg:ed].reshape(-1)
                top5 = torch.topk(pred, 5, dim=-1)[1]

                acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                acc_top5 = torch.eq(tar[:, None], top5).any(dim=1).float().mean().item() * 100
                ce = self.val_loss(pred, tar).item()
                std = pred.std(dim=-1).mean().item()
                norm = pred.norm(dim=-1).mean().item()

                stats = torch.tensor([acc, acc_top5, ce, std, norm], device=dist.get_device())
                dist.allreduce(stats)
                stats /= dist.get_world_size()
                acc, acc_top5, ce, std, norm = stats.tolist()

                logits_lg[f"logits_std_{self.resos[si]}"] = std
                logits_lg[f"logits_norm_{self.resos[si]}"] = norm
                kw[f"acc_{self.resos[si]}"] = acc
                kw[f"acc_top5_{self.resos[si]}"] = acc_top5
                kw[f"L_{self.resos[si]}"] = ce
                kw["acc_total"] += acc / len(self.begin_ends)
                kw["L_total"] += ce / len(self.begin_ends)

            if g_it % self.args.log_images_iters == 0:
                with FSDP.summon_full_params(self.switti, writeback=False):
                    torch.cuda.empty_cache()
                    for cfg in [0, 6]:
                        subprompt = prompt
                        imgs = self.pipe(
                            subprompt,
                            cfg=cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            return_pil=False,
                        )
                        imgs = make_grid(imgs, nrow=math.ceil(math.sqrt(len(imgs))))
                        tb_lg.log_image(
                            f"train_imgs_top_k={self.args.top_k}_top_p={self.args.top_p}_cfg={cfg}",
                            imgs,
                            step=g_it,
                        )

                        print("Eval prompts", self.eval_prompts)

                        imgs = self.pipe(
                            self.eval_prompts,
                            cfg=cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            return_pil=False,
                        )

                        # Add captions to images
                        captioned_imgs = []
                        for img, caption in zip(imgs, self.eval_prompts):
                            captioned_img = add_caption_to_image(img, caption)
                            captioned_imgs.append(captioned_img)

                        imgs = torch.stack(captioned_imgs)
                        imgs = make_grid(imgs, nrow=math.ceil(math.sqrt(len(imgs))))
                        tb_lg.log_image(
                            f"eval_imgs_topk={self.args.top_k}_top={self.args.top_p}_cfg={cfg}",
                            imgs,
                            step=g_it,
                        )

                        imgs = self.pipe(
                            self.eval_prompts,
                            top_k=1,
                            cfg=cfg,
                            return_pil=False,
                        )
                        imgs = make_grid(imgs, nrow=math.ceil(math.sqrt(len(imgs))))
                        tb_lg.log_image(f"eval_imgs_topk_1_cfg{cfg}", imgs, step=g_it)
                        del imgs

            if dist.is_master():
                tb_lg.update(head="Logits_stats", **logits_lg, step=g_it)
                tb_lg.update(head="AR_iter_loss", **kw, step=g_it)
            print(f"LOGGING {g_it} FINISHED")
            print(f"Step {g_it}, Loss: {kw['L_total']:.4f}")
            if self.args.use_gradient_checkpointing:
                self.switti.enable_gradient_checkpointing()
            self.switti.train()
            dist.barrier()

        return grad_norm.item(), scale_log2

    def get_config(self) -> Dict[str, Any]:
        """
        Get the training configuration.

        Returns:
            Dictionary containing training configuration parameters
        """
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
        }


class SwittiTrainerDisentanglement(SwittiTrainer):
    """
    Trainer class for Switti models with content-style disentanglement.

    This class extends the base trainer to support content-style disentanglement
    training with alternating loss weights and orthogonal loss constraints.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the disentanglement trainer.

        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments including training configuration
        """
        super().__init__(*args, **kwargs)
        self.placeholder_token_content_ids = self.placeholder_token_ids[0]
        self.placeholder_token_style_ids = self.placeholder_token_ids[1]

        self.placeholder_token_content_ids_2 = self.placeholder_token_ids_2[0]
        self.placeholder_token_style_ids_2 = self.placeholder_token_ids_2[1]

        self.major_scale_end = kwargs.get("args").major_scale_end
        self.major_type_scale = kwargs.get("args").major_type_scale
        self.minor_attend_weight = kwargs.get("args").minor_attend_weight

        self.current_status = 0
        self.alternative_step = 1

        if self.major_scale_end < 10:
            self.loss_weight_style = torch.zeros_like(self.loss_weight)
            self.loss_weight_content = torch.zeros_like(self.loss_weight)

            if self.major_type_scale == "style":
                style_scales = [scale for scale in range(self.major_scale_end)]
                content_scales = [
                    scale for scale in range(self.major_scale_end, len(self.begin_ends))
                ]
            else:
                style_scales = [
                    scale for scale in range(self.major_scale_end, len(self.begin_ends))
                ]
                content_scales = [scale for scale in range(self.major_scale_end)]

            normalize_style_factor = sum(
                self.patch_nums[int(scale)] * self.patch_nums[int(scale)] for scale in style_scales
            )
            normalize_content_factor = sum(
                self.patch_nums[int(scale)] * self.patch_nums[int(scale)]
                for scale in content_scales
            )

            cur_idx = 0
            for i_pn in range(len(self.patch_nums)):
                if i_pn in style_scales:
                    self.loss_weight_style[
                        :, cur_idx : cur_idx + self.patch_nums[i_pn] * self.patch_nums[i_pn]
                    ] = (1 / normalize_style_factor)
                elif i_pn in content_scales:
                    self.loss_weight_content[
                        :, cur_idx : cur_idx + self.patch_nums[i_pn] * self.patch_nums[i_pn]
                    ] = (1 / normalize_content_factor)
                cur_idx += self.patch_nums[i_pn] * self.patch_nums[i_pn]

            self.loss_weight_style[self.loss_weight_style == 0] = (
                self.minor_attend_weight * self.loss_weight_content[self.loss_weight_content != 0]
            )

            print("Loss weight style", self.loss_weight_style)
            print("Loss weight content", self.loss_weight_content)

        else:  # equal attends
            self.loss_weight_style = self.loss_weight
            self.loss_weight_content = self.loss_weight

        self.eval_prompts = []
        self.eval_prompts_content = []
        self.eval_prompts_style = []
        for i, prompt in enumerate(EVAL_PROMPTS_DISENTANGLEMENT["content"]):
            self.eval_prompts.append(
                prompt.format(kwargs.get("args").placeholder_token_content_use)
            )
            self.eval_prompts_content.append(
                "A {}".format(kwargs.get("args").placeholder_token_content_use)
            )
            self.eval_prompts_style.append("")
        for i, prompt in enumerate(EVAL_PROMPTS_DISENTANGLEMENT["style"]):
            self.eval_prompts.append(prompt.format(kwargs.get("args").placeholder_token_style_use))
            self.eval_prompts_content.append("")
            self.eval_prompts_style.append(
                "A {}".format(kwargs.get("args").placeholder_token_style_use)
            )
        for i, prompt in enumerate(EVAL_PROMPTS_DISENTANGLEMENT["content_style"]):
            self.eval_prompts.append(
                prompt.format(
                    kwargs.get("args").placeholder_token_content_use,
                    kwargs.get("args").placeholder_token_style_use,
                )
            )
            self.eval_prompts_content.append(
                "A {}".format(kwargs.get("args").placeholder_token_content_use)
            )
            self.eval_prompts_style.append(
                "A {}".format(kwargs.get("args").placeholder_token_style_use)
            )

        self.eval_prompts_content = self.eval_prompts
        self.eval_prompts_style = self.eval_prompts

    def loss_orthogonal(
        self, style_embeds: torch.Tensor, content_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute orthogonal loss between style and content embeddings.

        Args:
            style_embeds: Style embeddings tensor
            content_embeds: Content embeddings tensor

        Returns:
            Orthogonal loss value
        """
        # Normalize the embeddings
        style_norms = torch.norm(style_embeds, dim=1, keepdim=True)
        content_norms = torch.norm(content_embeds, dim=1, keepdim=True)

        # Normalize embeddings
        style_embeds_normalized = style_embeds / (style_norms + 1e-8)
        content_embeds_normalized = content_embeds / (content_norms + 1e-8)

        # Calculate cosine similarities
        cosine_similarities = torch.matmul(content_embeds_normalized, style_embeds_normalized.T)

        # Take absolute value and mean
        absolute_cosine_similarities = torch.abs(cosine_similarities)
        sum_cosine_similarities = torch.mean(absolute_cosine_similarities)

        return sum_cosine_similarities

    def adjust_loss_weight(self, is_train_style: int) -> None:
        """
        Adjust loss weights based on training mode.

        Args:
            is_train_style: Flag indicating whether to train style (1) or content (0)
        """
        if is_train_style == 1:
            self.loss_weight = self.loss_weight_style
        else:
            self.loss_weight = self.loss_weight_content

    def train_step(
        self,
        g_it: int,
        tb_lg: TensorboardLogger,
        orig_embeds_params: torch.Tensor,
        orig_embeds_params_2: torch.Tensor,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """
        Perform a single training step with disentanglement.

        Args:
            g_it: Global iteration number
            tb_lg: Tensorboard logger for metrics
            orig_embeds_params: Original embedding parameters for text encoder 1
            orig_embeds_params_2: Original embedding parameters for text encoder 2

        Returns:
            Tuple of (gradient_norm, scale_log2)
        """
        # forward
        self.pipe.text_encoder.transformer.train()
        self.pipe.text_encoder_2.transformer.train()

        self.current_status = 1 - self.current_status
        self.adjust_loss_weight(self.current_status)

        for accum_iter in range(self.grad_accum):
            image, prompt_content, prompt_style, prompt = next(self.dataloader)

            inp_B3HW = image.to(self.device, non_blocking=True)
            inp_B3HW = F.interpolate(
                inp_B3HW,
                size=(self.resos[-1], self.resos[-1]),
                mode="bicubic",
            )

            B, V = inp_B3HW.size(0), self.vae_local.vocab_size

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(
                inp_B3HW, noise_std=self.embed_noise_std
            )
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_switti_input(gt_idx_Bl)
            if self.args.uncond_proba > 0:
                cond_uncond_choice = torch.bernoulli(torch.full((B,), self.args.uncond_proba))
                for i_, p_ in enumerate(cond_uncond_choice):
                    if p_ == 1:
                        prompt[i_] = ""

            (
                prompt_embeds,
                pooled_prompt_embeds,
                prompt_attn_bias,
            ) = self.pipe.encode_prompt(prompt, encode_null=False, **self.pipe.kwargs_bonus)

            prompt_content_embeds, pooled_prompt_content_embeds, prompt_content_attn_bias = (
                self.pipe.encode_prompt(prompt_content, encode_null=False, **self.pipe.kwargs_bonus)
            )
            prompt_style_embeds, pooled_prompt_style_embeds, prompt_style_attn_bias = (
                self.pipe.encode_prompt(prompt_style, encode_null=False, **self.pipe.kwargs_bonus)
            )

            with self.optimizer.amp_ctx:
                batch_embed = prompt_embeds.shape[0]

                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    batch_height=batch_embed * [self.resos[-1]],
                    batch_width=batch_embed * [self.resos[-1]],
                    prompt_content_embeds=prompt_content_embeds,
                    prompt_content_attn_bias=prompt_content_attn_bias,
                    prompt_style_embeds=prompt_style_embeds,
                    prompt_style_attn_bias=prompt_style_attn_bias,
                    is_train_style=self.current_status,
                )

                loss = self.train_loss(
                    logits_BLV.view(-1, V),
                    gt_BL.view(-1),
                ).view(B, -1)
                loss = loss.mul(self.loss_weight).sum(dim=-1).mean()

                content_embeds = self.pipe.text_encoder.transformer.get_input_embeddings().weight[
                    self.placeholder_token_content_ids
                ]
                content_embeds_2 = (
                    self.pipe.text_encoder_2.transformer.get_input_embeddings().weight[
                        self.placeholder_token_content_ids_2
                    ]
                )
                style_embeds = self.pipe.text_encoder.transformer.get_input_embeddings().weight[
                    self.placeholder_token_style_ids
                ]
                style_embeds_2 = self.pipe.text_encoder_2.transformer.get_input_embeddings().weight[
                    self.placeholder_token_style_ids_2
                ]

                if self.args.enable_orthogonal_loss:
                    loss_orthogonal = self.loss_orthogonal(
                        style_embeds, content_embeds
                    ) + self.loss_orthogonal(style_embeds_2, content_embeds_2)
                    loss += self.args.orthogonal_loss_weight * loss_orthogonal.mean()

            # backward
            is_stepping = (accum_iter + 1) == self.grad_accum
            grad_norm, scale_log2 = self.optimizer.backward_clip_step(
                loss=loss,
                is_stepping=is_stepping,
            )

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones(
                (len(self.pipe.text_encoder.tokenizer),), dtype=torch.bool
            )
            index_no_updates[
                min(self.placeholder_token_content_ids) : max(self.placeholder_token_content_ids)
                + 1
            ] = False
            index_no_updates[
                min(self.placeholder_token_style_ids) : max(self.placeholder_token_style_ids) + 1
            ] = False

            index_no_updates_2 = torch.ones(
                (len(self.pipe.text_encoder_2.tokenizer),), dtype=torch.bool
            )
            index_no_updates_2[
                min(self.placeholder_token_content_ids_2) : max(
                    self.placeholder_token_content_ids_2
                )
                + 1
            ] = False
            index_no_updates_2[
                min(self.placeholder_token_style_ids_2) : max(self.placeholder_token_style_ids_2)
                + 1
            ] = False

            with torch.no_grad():
                self.pipe.text_encoder.transformer.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]
                self.pipe.text_encoder_2.transformer.get_input_embeddings().weight[
                    index_no_updates_2
                ] = orig_embeds_params_2[index_no_updates_2]

        # log to tensorboard
        if g_it > 0 and g_it % self.args.log_iters == 0:
            # recalculate logits in .eval() mode to log acc
            self.switti.eval()
            if self.args.use_gradient_checkpointing:
                self.switti.disable_gradient_checkpointing()
            with torch.no_grad(), self.optimizer.amp_ctx:
                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    batch_height=batch_embed * [self.resos[-1]],
                    batch_width=batch_embed * [self.resos[-1]],
                    prompt_content_embeds=prompt_content_embeds,
                    prompt_content_attn_bias=prompt_content_attn_bias,
                    prompt_style_embeds=prompt_style_embeds,
                    prompt_style_attn_bias=prompt_style_attn_bias,
                    is_train_style=self.current_status,
                )

            # Compute cluster usage
            pred_BL = logits_BLV.data.argmax(dim=-1)
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float().cuda()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100

            logits_lg = dict()
            kw = dict(z_voc_usage=cluster_usage, acc_total=0.0, L_total=0.0)
            for si, (bg, ed) in enumerate(self.begin_ends):
                pred = logits_BLV.data[:, bg:ed].reshape(-1, V)
                tar = gt_BL[:, bg:ed].reshape(-1)
                top5 = torch.topk(pred, 5, dim=-1)[1]

                acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                acc_top5 = torch.eq(tar[:, None], top5).any(dim=1).float().mean().item() * 100
                ce = self.val_loss(pred, tar).item()
                std = pred.std(dim=-1).mean().item()
                norm = pred.norm(dim=-1).mean().item()

                stats = torch.tensor([acc, acc_top5, ce, std, norm], device=dist.get_device())
                dist.allreduce(stats)
                stats /= dist.get_world_size()
                acc, acc_top5, ce, std, norm = stats.tolist()

                logits_lg[f"logits_std_{self.resos[si]}"] = std
                logits_lg[f"logits_norm_{self.resos[si]}"] = norm
                kw[f"acc_{self.resos[si]}"] = acc
                kw[f"acc_top5_{self.resos[si]}"] = acc_top5
                kw[f"L_{self.resos[si]}"] = ce
                kw["acc_total"] += acc / len(self.begin_ends)
                kw["L_total"] += ce / len(self.begin_ends)

            enable = False
            if enable and g_it % self.args.log_images_iters == 0:
                with FSDP.summon_full_params(self.switti, writeback=False):
                    torch.cuda.empty_cache()
                    for cfg in [0, 6]:
                        subprompt = prompt
                        imgs = self.pipe(
                            prompt,
                            prompt_content,
                            prompt_style,
                            cfg=cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            return_pil=False,
                        )
                        imgs = make_grid(imgs, nrow=math.ceil(math.sqrt(len(imgs))))
                        tb_lg.log_image(
                            f"train_imgs_top_k={self.args.top_k}_top_p={self.args.top_p}_cfg={cfg}",
                            imgs,
                            step=g_it,
                        )

            if dist.is_master():
                tb_lg.update(head="Logits_stats", **logits_lg, step=g_it)
                tb_lg.update(head="AR_iter_loss", **kw, step=g_it)
            print(f"LOGGING {g_it} FINISHED")
            print(f"Step {g_it}, Loss: {kw['L_total']:.4f}")
            if self.args.use_gradient_checkpointing:
                self.switti.enable_gradient_checkpointing()
            self.switti.train()
            dist.barrier()

        return grad_norm.item(), scale_log2


def add_caption_to_image(image: torch.Tensor, caption: str, font_size: int = 20) -> torch.Tensor:
    """
    Add caption text to an image tensor.

    Args:
        image: Input image tensor
        caption: Text caption to add
        font_size: Font size for the caption (default: 20)

    Returns:
        Image tensor with caption added
    """
    # Convert tensor to PIL Image if needed
    if isinstance(image, torch.Tensor):
        image = (image * 255).clamp(0, 255).to(torch.uint8)
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image)

    # Create new image with space for caption
    margin = 10
    width = image.width
    height = image.height + font_size + 2 * margin
    new_image = Image.new("RGB", (width, height), "white")
    new_image.paste(image, (0, 0))

    # Add caption
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Center the text
    text_width = draw.textlength(caption, font=font)
    x = (width - text_width) // 2
    y = height - font_size - margin

    draw.text((x, y), caption, fill="black", font=font)

    # Convert back to tensor
    new_image = torch.from_numpy(np.array(new_image)).permute(2, 0, 1).float() / 255.0
    return new_image
