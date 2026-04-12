
# Code adapted from:
# https://github.com/yandex-research/switti/blob/master/models/pipeline.py

import math
import os

import safetensors
import torch
from PIL.Image import Image as PILImage
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from models.clip import FrozenCLIPEmbedder
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.switti import SwittiHF, get_crop_condition
from models.vqvae import VQVAEHF


class SwittiPipeline:
    vae_path = "yresearch/VQVAE-Switti"
    text_encoder_path = "openai/clip-vit-large-patch14"
    text_encoder_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    def __init__(
        self,
        switti,
        vae,
        text_encoder,
        text_encoder_2,
        device,
        dtype=torch.float32,
    ):
        self.switti = switti.to(dtype)
        self.vae = vae.to(dtype)
        self.text_encoder = text_encoder.to(dtype)
        self.text_encoder_2 = text_encoder_2.to(dtype)

        self.switti.eval()
        self.vae.eval()

        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
        prompt_config=None,
        **kwargs,
    ):
        switti = SwittiHF.from_pretrained(
            pretrained_model_name_or_path,
            prompt_config=prompt_config,
        ).to(device)
        vae = VQVAEHF.from_pretrained(cls.vae_path).to(device)
        text_encoder = FrozenCLIPEmbedder(cls.text_encoder_path, device=device)
        text_encoder_2 = FrozenCLIPEmbedder(cls.text_encoder_2_path, device=device)

        return cls(switti, vae, text_encoder, text_encoder_2, device, torch_dtype, **kwargs)

    @staticmethod
    def to_image(tensor):
        return [ToPILImage()((255 * img.cpu().detach()).to(torch.uint8)) for img in tensor]

    def _encode_prompt(self, prompt: str | list[str], **kwargs):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        encodings = [
            self.text_encoder.encode(prompt, **kwargs),
            self.text_encoder_2.encode(prompt, **kwargs),
        ]
        prompt_embeds = torch.concat([encoding.last_hidden_state for encoding in encodings], dim=-1)
        pooled_prompt_embeds = encodings[-1].pooler_output
        attn_bias = encodings[-1].attn_bias

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def _encode_prompt_residual(
        self,
        prompt: str | list[str],
        residual_tokens: torch.Tensor,
        residual_tokens_2: torch.Tensor,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # tokenize prompt
        batch_encoding = self.text_encoder.tokenizer(
            prompt,
            truncation=True,
            max_length=self.text_encoder.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).to(self.text_encoder.device)

        # encode prompt

        encodings = [
            self.text_encoder.encode(prompt),
            self.text_encoder_2.encode(prompt),
        ]
        pooled_prompt_embeds = encodings[-1].pooler_output
        attn_bias = encodings[-1].attn_bias

        # create a zero tensor with the same shape as prompt_embeds
        residual_embeds_1 = torch.zeros_like(encodings[0].last_hidden_state)
        residual_embeds_2 = torch.zeros_like(encodings[1].last_hidden_state)

        # add residual tokens to where the tokens are in residual tokens.key()
        for token_id in residual_tokens.keys():
            residual_embeds_1[batch_encoding.input_ids == int(token_id)] = residual_tokens[token_id]

        for token_id in residual_tokens_2.keys():
            residual_embeds_2[batch_encoding.input_ids == int(token_id)] = residual_tokens_2[
                token_id
            ]

        encodings_0 = encodings[0].last_hidden_state + residual_embeds_1
        encodings_1 = encodings[1].last_hidden_state + residual_embeds_2

        prompt_embeds = torch.concat([encodings_0, encodings_1], dim=-1)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def _encode_prompt_with_appending_residual_tokens(
        self,
        prompt: str | list[str],
        residual_tokens: torch.Tensor,
        residual_tokens_2: torch.Tensor,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # tokenize prompt
        batch_encoding = self.text_encoder.tokenizer(
            prompt,
            truncation=True,
            max_length=self.text_encoder.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).to(self.text_encoder.device)

        # encode prompt

        encodings = [
            self.text_encoder.encode(prompt),
            self.text_encoder_2.encode(prompt),
        ]
        pooled_prompt_embeds = encodings[-1].pooler_output
        attn_bias = encodings[-1].attn_bias

        # residual_tokens is nn.ParameterDict, then I need to convert it to a tensor
        expand_dim = encodings[0].last_hidden_state.shape[0]
        residual_tokens_tensor = torch.stack(
            [residual_tokens[token_id] for token_id in residual_tokens.keys()]
        ).expand(expand_dim, -1, -1)
        residual_tokens_2_tensor = torch.stack(
            [residual_tokens_2[token_id] for token_id in residual_tokens_2.keys()]
        ).expand(expand_dim, -1, -1)

        encodings_0 = torch.cat([residual_tokens_tensor, encodings[0].last_hidden_state], dim=1)
        encodings_1 = torch.cat([residual_tokens_2_tensor, encodings[1].last_hidden_state], dim=1)

        prompt_embeds = torch.concat([encodings_0, encodings_1], dim=-1)

        attn_bias_added = torch.zeros(attn_bias.shape[0], residual_tokens_tensor.shape[1]).to(
            attn_bias.device
        )

        attn_bias = torch.cat([attn_bias_added, attn_bias], dim=1)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        encode_null: bool = True,
        **kwargs,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt(prompt, **kwargs)
        if encode_null:
            B, L, hidden_dim = prompt_embeds.shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(
                null_prompt, **kwargs
            )

            null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(
                pooled_prompt_embeds.device
            )
            null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

            prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt_residual(
        self,
        prompt: str | list[str],
        residual_tokens: torch.Tensor,
        residual_tokens_2: torch.Tensor,
        null_prompt: str = "",
        encode_null: bool = True,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt_residual(
            prompt, residual_tokens, residual_tokens_2
        )

        if encode_null:
            B, L, hidden_dim = prompt_embeds.shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)

            null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(
                pooled_prompt_embeds.device
            )
            null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

            prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt_with_appending_residual_tokens(
        self,
        prompt: str | list[str],
        residual_tokens: torch.Tensor,
        residual_tokens_2: torch.Tensor,
        null_prompt: str = "",
        encode_null: bool = True,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = (
            self._encode_prompt_with_appending_residual_tokens(
                prompt, residual_tokens, residual_tokens_2
            )
        )

        if encode_null:
            B, L, hidden_dim = prompt_embeds.shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)

            # shape of null_embeds
            L_null = null_embeds.shape[1]
            # concat first L - L_null of prompt_embeds to null_embeds
            null_embeds = torch.cat(
                [prompt_embeds[0, : L - L_null, :].unsqueeze(0), null_embeds], dim=1
            )
            null_attn_bias = torch.cat(
                [attn_bias[0, : L - L_null].unsqueeze(0), null_attn_bias], dim=1
            )

            null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(
                pooled_prompt_embeds.device
            )
            null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

            prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        seed: int | None = None,
        cfg: float = 6.0,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        image_size: tuple[int, int] = (512, 512),
        last_scale_temp: None | float = None,
        return_intermediate_scaled_image: bool = False,
    ) -> torch.Tensor | list[PILImage]:
        """
        only used for inference, on autoregressive mode
        :param prompt: text prompt to generate an image
        :param null_prompt: negative prompt for CFG
        :param seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
        """
        assert not self.switti.training
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        if seed is None:
            rng = None
        else:
            switti.rng.manual_seed(seed)
            rng = switti.rng

        image_each_token_map = []

        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)

        B = context.shape[0] // 2

        cond_vector = switti.text_pooler(cond_vector)

        if switti.use_crop_cond:
            crop_coords = get_crop_condition(
                2 * B * [image_size[0]],
                2 * B * [image_size[1]],
            ).to(cond_vector.device)
            crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
            crop_cond = switti.crop_proj(crop_embed)
        else:
            crop_cond = None

        sos = cond_BD = cond_vector

        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        if not switti.rope:
            lvl_pos += switti.pos_1LC
        next_token_map = (
            sos.unsqueeze(1)
            + switti.pos_start.expand(2 * B, switti.first_l, -1)
            + lvl_pos[:, : switti.first_l]
        )
        cur_L = 0
        f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])

        for b in switti.blocks:
            b.attn.kv_caching(switti.use_ar)  # Use KV caching if switti is in the AR mode
            b.cross_attn.kv_caching(True)

        for si, pn in enumerate(switti.patch_nums):  # si: i-th segment
            ratio = si / switti.num_stages_minus_1
            x_BLC = next_token_map

            if switti.rope:
                freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = switti.freqs_cis

            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                context = context[:B]
                context_attn_bias = context_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                for b in switti.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
            else:
                apply_smooth = more_smooth

            for block in switti.blocks:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
            cur_L += pn * pn

            logits_BlV = switti.get_logits(x_BLC, cond_BD)

            # Guidance
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio),
                    tau=gum_t,
                    hard=False,
                    dim=-1,
                    rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                # default nucleus sampling
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV,
                    rng=rng,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                si,
                len(switti.patch_nums),
                f_hat,
                h_BChw,
            )
            image_each_token_map.append(f_hat.clone())

            if si != switti.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    switti.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                )
                # double the batch sizes due to CFG
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

        # de-normalize, from [-1, 1] to [0, 1]
        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        batch_size = img.shape[0]

        if return_pil:
            img = self.to_image(img)

        if return_intermediate_scaled_image:

            batch_image_intermediate_scaled_image = []
            for i in range(batch_size):
                idx_image_each_token_map = [
                    vae.fhat_to_img(inter_f[i].unsqueeze(0)).add(1).mul(0.5).squeeze(0)
                    for inter_f in image_each_token_map
                ]
                grid_image = make_grid(
                    idx_image_each_token_map,
                    nrow=math.ceil(math.sqrt(len(idx_image_each_token_map))),
                )
                batch_image_intermediate_scaled_image.append(grid_image)

            batch_image_intermediate_scaled_image = self.to_image(
                torch.stack(batch_image_intermediate_scaled_image)
            )

            return img, batch_image_intermediate_scaled_image

        return img

    def load_textual_inversion_state_dicts(self, pretrained_model_name_or_paths, **kwargs):
        state_dicts = []
        for pretrained_model_name_or_path in pretrained_model_name_or_paths:
            state_dict = safetensors.torch.load_file(pretrained_model_name_or_path, device="cpu")
            state_dicts.append(state_dict)
        return state_dicts

    def load_textual_inversion(
        self,
        pretrained_model_name_or_path,
        token=None,
        tokenizer=None,  # noqa: F821
        text_encoder=None,  # noqa: F821
        **kwargs,
    ):

        tokenizer = tokenizer
        text_encoder = text_encoder

        # 2. Normalize inputs
        pretrained_model_name_or_paths = (
            [pretrained_model_name_or_path]
            if not isinstance(pretrained_model_name_or_path, list)
            else pretrained_model_name_or_path
        )
        tokens = [token] if not isinstance(token, list) else token
        if tokens[0] is None:
            tokens = tokens * len(pretrained_model_name_or_paths)

        # 4. Load state dicts of textual embeddings
        state_dicts = self.load_textual_inversion_state_dicts(
            pretrained_model_name_or_paths, **kwargs
        )

        # 4.1 Handle the special case when state_dict is a tensor that contains n embeddings for n tokens
        if len(tokens) > 1 and len(state_dicts) == 1:
            if isinstance(state_dicts[0], torch.Tensor):
                state_dicts = list(state_dicts[0])
                if len(tokens) != len(state_dicts):
                    raise ValueError(
                        f"You have passed a state_dict contains {len(state_dicts)} embeddings, and list of tokens of length {len(tokens)} "
                        f"Make sure both have the same length."
                    )

        # 4. Retrieve tokens and embeddings
        tokens, embeddings = self._retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer)

        # 5. Extend tokens and embeddings for multi vector
        tokens, embeddings = self._extend_tokens_and_embeddings(tokens, embeddings, tokenizer)

        # 7.2 save expected device and dtype
        device = text_encoder.device
        dtype = text_encoder.dtype

        # 7.3 Increase token embedding matrix
        text_encoder.resize_token_embeddings(len(tokenizer) + len(tokens))
        input_embeddings = text_encoder.get_input_embeddings().weight

        print("added", tokens)
        # 7.4 Load token and embedding
        added_tokens_id = []
        for token, embedding in zip(tokens, embeddings):
            # add tokens and get ids
            tokenizer.add_tokens(token)
            token_id = tokenizer.convert_tokens_to_ids(token)
            input_embeddings.data[token_id] = embedding
            added_tokens_id.append(token_id)
        input_embeddings.to(dtype=dtype, device=device)

        # ids of added tokens
        return added_tokens_id

    def _retrieve_tokens_and_embeddings(self, tokens, state_dicts, tokenizer):
        all_tokens = []
        all_embeddings = []
        for state_dict, token in zip(state_dicts, tokens):
            if isinstance(state_dict, torch.Tensor):
                if token is None:
                    raise ValueError(
                        "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                    )
                loaded_token = token
                embedding = state_dict
            elif len(state_dict) == 1:
                # diffusers
                loaded_token, embedding = next(iter(state_dict.items()))
            elif "string_to_param" in state_dict:
                # A1111
                loaded_token = state_dict["name"]
                embedding = state_dict["string_to_param"]["*"]
            else:
                raise ValueError(
                    f"Loaded state dictionary is incorrect: {state_dict}. \n\n"
                    "Please verify that the loaded state dictionary of the textual embedding either only has a single key or includes the `string_to_param`"
                    " input key."
                )

            if token is not None and loaded_token != token:
                print(
                    f"The loaded token: {loaded_token} is overwritten by the passed token {token}."
                )
            else:
                token = loaded_token

            if token in tokenizer.get_vocab():
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )

            all_tokens.append(token)
            all_embeddings.append(embedding)

        return all_tokens, all_embeddings

    def _extend_tokens_and_embeddings(self, tokens, embeddings, tokenizer):
        all_tokens = []
        all_embeddings = []

        for embedding, token in zip(embeddings, tokens):
            if f"{token}_1" in tokenizer.get_vocab():
                multi_vector_tokens = [token]
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    multi_vector_tokens.append(f"{token}_{i}")
                    i += 1

                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )

            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1
            if is_multi_vector:
                all_tokens += [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                all_embeddings += [e for e in embedding]  # noqa: C416
            else:
                all_tokens += [token]
                all_embeddings += [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        return all_tokens, all_embeddings


class SwittiDisentangledPipeline(SwittiPipeline):
    def __init__(
        self, switti, vae, text_encoder, text_encoder_2, device, dtype=torch.float32, **kwargs
    ):
        super().__init__(switti, vae, text_encoder, text_encoder_2, device, dtype)
        self.proj_path = kwargs.get("proj_path", None)
        self.proj_2_path = kwargs.get("proj_2_path", None)

        if self.proj_path is not None and os.path.exists(self.proj_path):
            self.proj = torch.load(self.proj_path)
        else:
            self.proj = None
        if self.proj_2_path is not None and os.path.exists(self.proj_2_path):
            self.proj_2 = torch.load(self.proj_2_path)
        else:
            self.proj_2 = None

        self.kwargs_bonus = {
            "proj_matrix": self.proj,
            "proj_matrix_2": self.proj_2,
            "placeholder_token_content_ids": None,
            "placeholder_token_content_ids_2": None,
            "placeholder_token_style_ids": None,
            "placeholder_token_style_ids_2": None,
        }

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        prompt_content: str | list[str],
        prompt_style: str | list[str],
        null_prompt: str = "",
        seed: int | None = None,
        cfg: float = 6.0,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        image_size: tuple[int, int] = (512, 512),
        last_scale_temp: None | float = None,
        return_intermediate_scaled_image: bool = False,
        style_scale_end: int = 3,
        mode: str = "content_style",
    ) -> torch.Tensor | list[PILImage]:
        """
        Inherits from SwittiPipeline.__call__ and switches prompts from prompt1 to prompt2 at a specified scale.

        :param prompt1: First text prompt to generate the initial scales of the image.
        :param prompt2: Second text prompt to generate the remaining scales of the image.
        :param switch_scale: The scale index at which to switch from prompt1 to prompt2.
                             If None, it switches halfway through the scales.
        :param Other parameters are the same as in SwittiPipeline.__call__.
        :return: Generated image(s) as PIL Images or torch tensors, optionally with intermediate scaled images.
        """
        assert not self.switti.training
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        if seed is None:
            rng = None
        else:
            switti.rng.manual_seed(seed)
            rng = switti.rng

        infer_selected_scales = switti.selected_scales

        if len(infer_selected_scales) == 2:  # divie as group content and style
            if mode == "content":
                infer_selected_scales = [switti.selected_scales[1]]
            elif mode == "style":
                infer_selected_scales = [switti.selected_scales[0]]

        print("Mode: ", mode)

        image_each_token_map = []

        # Encode both prompts
        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, **self.kwargs_bonus)
        context_content, cond_vector_content, context_attn_bias_content = self.encode_prompt(
            prompt_content, **self.kwargs_bonus
        )
        context_style, cond_vector_style, context_attn_bias_style = self.encode_prompt(
            prompt_style, **self.kwargs_bonus
        )

        B = context.shape[0] // 2

        cond_vector = switti.text_pooler(cond_vector)
        cond_vector_content = switti.text_pooler(cond_vector_content)
        cond_vector_style = switti.text_pooler(cond_vector_style)

        # Determine when to switch prompts
        total_scales = len(switti.patch_nums)

        # Prepare initial conditions
        current_context = context
        current_cond_vector = cond_vector
        current_attn_bias = context_attn_bias

        if switti.use_crop_cond:
            crop_coords = get_crop_condition(
                2 * B * [image_size[0]],
                2 * B * [image_size[1]],
            ).to(current_cond_vector.device)
            crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
            crop_cond = switti.crop_proj(crop_embed)
        else:
            crop_cond = None

        sos = cond_BD = current_cond_vector

        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        if not switti.rope:
            lvl_pos += switti.pos_1LC
        next_token_map = (
            sos.unsqueeze(1)
            + switti.pos_start.expand(2 * B, switti.first_l, -1)
            + lvl_pos[:, : switti.first_l]
        )
        cur_L = 0
        f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])

        mode = None
        for b in switti.blocks:
            b.attn.kv_caching(switti.use_ar)
            b.cross_attn.kv_caching(True)

        virtual_tokens = []
        for i, pn in enumerate(switti.patch_nums):
            virtual_token_idx = torch.full(
                (1, switti.num_virtual_tokens_per_scale),
                i * switti.num_virtual_tokens_per_scale,
                dtype=torch.long,
                device=f_hat.device,
            )
            virtual_tokens.append(self.switti.virtual_token_embed(virtual_token_idx))

        insert_selected_blocks = range(switti.num_insert_blocks)

        for si, pn in enumerate(switti.patch_nums):  # si: scale index
            ratio = si / switti.num_stages_minus_1
            x_BLC = next_token_map

            if si <= style_scale_end:
                mode = 1
                current_context = context_style
                current_cond_vector = cond_vector_style
                current_attn_bias = context_attn_bias_style

            else:
                if mode == 1:
                    for b in switti.blocks:
                        b.cross_attn.kv_caching(True)
                    mode = 2
                current_context = context_content
                current_cond_vector = cond_vector_content
                current_attn_bias = context_attn_bias_content

            if switti.rope:
                freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = switti.freqs_cis

            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                current_context = current_context[:B]
                current_attn_bias = current_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                for b in switti.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
            else:
                apply_smooth = more_smooth

            for idx_block, block in enumerate(switti.blocks):

                query_input = x_BLC.clone()

                expanded_virtual_tokens_list = []
                for i in range(len(virtual_tokens)):
                    start_idx = idx_block * switti.C * 2
                    end_idx = (idx_block + 1) * switti.C * 2
                    expanded_virtual_tokens = virtual_tokens[i].expand(x_BLC.size(0), -1, -1)[
                        :, :, start_idx:end_idx
                    ]
                    expanded_virtual_tokens_list.append(expanded_virtual_tokens)

                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=current_context,
                    context_attn_bias=current_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                    virtual_kv=(
                        expanded_virtual_tokens_list
                        if (si in infer_selected_scales) and (idx_block in insert_selected_blocks)
                        else None
                    ),
                    non_virtual_indices=switti.non_virtual_indices,
                    selected_scales=infer_selected_scales,
                )

            cur_L += pn * pn

            logits_BlV = switti.get_logits(x_BLC, cond_BD)

            # Guidance
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio),
                    tau=gum_t,
                    hard=False,
                    dim=-1,
                    rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV,
                    rng=rng,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                si,
                len(switti.patch_nums),
                f_hat,
                h_BChw,
            )
            image_each_token_map.append(f_hat.clone())

            if si != switti.num_stages_minus_1:
                next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    switti.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                )
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        batch_size = img.shape[0]

        if return_pil:
            img = self.to_image(img)

        if return_intermediate_scaled_image:
            batch_image_intermediate_scaled_image = []
            for i in range(batch_size):
                idx_image_each_token_map = [
                    vae.fhat_to_img(inter_f[i].unsqueeze(0)).add(1).mul(0.5).squeeze(0)
                    for inter_f in image_each_token_map
                ]
                grid_image = make_grid(
                    idx_image_each_token_map,
                    nrow=math.ceil(math.sqrt(len(idx_image_each_token_map))),
                )
                batch_image_intermediate_scaled_image.append(grid_image)

            batch_image_intermediate_scaled_image = self.to_image(
                torch.stack(batch_image_intermediate_scaled_image)
            )

            return img, batch_image_intermediate_scaled_image

        return img
