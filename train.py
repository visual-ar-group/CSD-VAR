
"""
Training script for textual inversion disentanglement.

This script implements the training pipeline for Switti models with content-style
disentanglement capabilities, supporting multi-vector textual inversion and
virtual token embeddings for enhanced generation control.
"""

import gc
import os
import sys
import time
from typing import Dict, Tuple

import torch
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader

import dist
from models import VQVAE, VQVAEHF, Switti, build_models_disentangled
from models.basic_switti import AdaLNSelfCrossAttn
from trainer import SwittiTrainerDisentanglement
from utils import arg_util, misc
from utils.amp_sc import AmpOptimizer
from utils.data import (
    build_personalized_dataset_disentanglement,
)
from utils.data_sampler import DistInfiniteBatchSampler
from utils.fsdp import (
    load_model_state,
    save_progress,
    save_training_config,
)
from utils.lr_control import filter_params, lr_wd_annealing


def build_everything(
    args: arg_util.Args,
) -> Tuple[misc.DistLogger, SwittiTrainerDisentanglement, int]:
    """
    Build and configure all components for training.

    This function initializes the complete training pipeline including models,
    optimizers, data loaders, and trainers with proper configuration.

    Args:
        args: Training arguments and configuration

    Returns:
        Tuple containing (tensorboard_logger, trainer, start_iteration)

    Raises:
        ValueError: If prompt config file doesn't exist or invalid configuration
    """
    # Load prompt config
    if not os.path.exists(args.prompt_config_path):
        raise ValueError(f"Prompt config file {args.prompt_config_path} does not exist")

    with open(args.prompt_config_path, "r") as f:
        prompt_config = yaml.safe_load(f)

    print("\n=== Prompt Configuration ===")
    print(f"Loaded prompt config from: {args.prompt_config_path}")
    print("Configuration details:")
    for key, value in prompt_config.items():
        print(f"{key}: {value}")
    print("===========================\n")

    # Save configs to output directory
    if dist.is_master():
        # Create experiment directory
        exp_dir = os.path.join(args.local_out_dir_path, "config")
        os.makedirs(exp_dir, exist_ok=True)

        # Save args in a cleaner format
        args_save_path = os.path.join(exp_dir, "args_config.yaml")
        args_dict = {}
        for key, value in args.state_dict().items():
            # Skip None values and empty strings
            if value is not None and value != "":
                # Convert tuples to lists for better YAML formatting
                if isinstance(value, tuple):
                    value = list(value)
                args_dict[key] = value

        with open(args_save_path, "w") as f:
            yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Saved args config to: {args_save_path}")

        # Save prompt config with custom formatting
        prompt_config_save_path = os.path.join(exp_dir, "prompt_config.yaml")
        with open(prompt_config_save_path, "w") as f:
            # Make a copy of prompt_config for saving
            prompt_config_save = prompt_config.copy()
            # Custom formatting for selected_scales
            selected_scales = prompt_config_save.pop("selected_scales")
            f.write(f"selected_scales: {selected_scales}\n")
            # Write remaining config
            yaml.dump(prompt_config_save, f, default_flow_style=False)
        print(f"Saved prompt config to: {prompt_config_save_path}")

    # check if use_ar is True, path pretrained_path is yresearch/Switti-AR
    if args.use_ar and args.pretrained_path != "yresearch/Switti-AR":
        raise ValueError("use_ar is True, but pretrained_path is not yresearch/Switti-AR")

    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    if dist.is_master():
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_path,
                filename_suffix=f'__{misc.time_str("%m%d_%H%M")}',
            ),
            verbose=True,
        )
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)

    # log args
    print(f"initial args:\n{str(args)}")

    # build models
    vae_local, switti_wo_ddp, pipe = build_models_disentangled(
        # VQVAE hyperparameters
        V=args.vqvae_vocab_size,
        Cvae=args.vqvae_channel_dim,
        ch=args.vqvae_n_channels,
        share_quant_resi=args.vqvae_share_quant_resi,
        # train hyperparameters
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        depth=args.depth,
        attn_l2_norm=args.anorm,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
        text_encoder_path=args.text_encoder_path,
        text_encoder_2_path=args.text_encoder_2_path,
        pretrained_path=args.pretrained_path,
        use_ar=args.use_ar,
        rope=args.rope,
        rope_theta=args.rope_theta,
        rope_size=args.rope_size,
        dpr=args.drop_path_rate,
        use_swiglu_ffn=args.use_swiglu_ffn,
        use_crop_cond=args.use_crop_cond,
        freeze_text_encoder=False,
        num_virtual_tokens_per_scale=args.num_virtual_tokens_per_scale,
        prompt_config=prompt_config,
        proj_path=args.proj_path,
        proj_2_path=args.proj_2_path,
    )
    # Load VAE and Switti checkpoints

    # if noise raise error
    if args.vae_ckpt is None:
        raise ValueError("vae_ckpt is None")

    vae_local = VQVAEHF.from_pretrained(args.vae_ckpt).to(dist.get_device())

    start_it = load_model_state(args, switti_wo_ddp)
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    switti_wo_ddp: Switti = args.compile_model(switti_wo_ddp, args.tfast)
    if args.use_gradient_checkpointing:
        switti_wo_ddp.enable_gradient_checkpointing()

    print(f"[INIT] Switti model = {switti_wo_ddp}\n\n")
    count_p = lambda m: f"{sum(p.numel() for p in m.parameters())/1e6:.2f}"
    print(
        "[INIT][#para] "
        + ", ".join(
            [
                f"{k}={count_p(m)}"
                for k, m in (
                    ("VAE", vae_local),
                    ("VAE.enc", vae_local.encoder),
                    ("VAE.dec", vae_local.decoder),
                    ("VAE.quant", vae_local.quantize),
                )
            ]
        )
    )
    print(
        "[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}" for k, m in (("Switti", switti_wo_ddp),)])
        + "\n\n"
    )

    print("\n=== Model Parameters and Memory Usage ===")
    print(f"VAE Parameters: {sum(p.numel() for p in vae_local.parameters()):,}")
    print(f"Switti Parameters: {sum(p.numel() for p in switti_wo_ddp.parameters()):,}")
    print(f"Text Encoder 1 Parameters: {sum(p.numel() for p in pipe.text_encoder.parameters()):,}")
    print(
        f"Text Encoder 2 Parameters: {sum(p.numel() for p in pipe.text_encoder_2.parameters()):,}"
    )

    print("\nApproximate Model Memory Usage:")
    print(f"VAE Memory: {get_model_size(vae_local):.2f} MB")
    print(f"Switti Memory: {get_model_size(switti_wo_ddp):.2f} MB")
    print(f"Text Encoder 1 Memory: {get_model_size(pipe.text_encoder):.2f} MB")
    print(f"Text Encoder 2 Memory: {get_model_size(pipe.text_encoder_2):.2f} MB")

    # Get current GPU memory usage
    if torch.cuda.is_available():
        print("\nCurrent GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        print("\nTrainable Parameters:")
        trainable_params = sum(p.numel() for p in pipe.text_encoder.parameters() if p.requires_grad)
        trainable_params += sum(
            p.numel() for p in pipe.text_encoder_2.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {trainable_params:,}")
        print("=====================================\n")

    print("\nEstimated Training Memory Usage (per batch):")
    for model_name, model in [
        ("Text Encoder 1", pipe.text_encoder),
        ("Text Encoder 2", pipe.text_encoder_2),
    ]:
        train_mem = estimate_training_memory(model, args.batch_size)
        print(f"\n{model_name}:")
        print(f"Parameters Memory: {train_mem['parameters']:.2f} MB")
        print(f"Gradients Memory: {train_mem['gradients']:.2f} MB")
        print(f"Optimizer States Memory: {train_mem['optimizer_states']:.2f} MB")
        print(f"Forward Activation Memory: {train_mem['activation_forward']:.2f} MB")
        print(f"Backward Activation Memory: {train_mem['activation_backward']:.2f} MB")
        print(f"Total Training Memory: {train_mem['total']:.2f} MB")

    # Add peak memory tracking during actual training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("\nPeak GPU Memory During Training:")
        print(f"Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"Peak Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")

    # FSDP wrapper
    switti: FSDP = (FSDP if dist.initialized() else NullDDP)(
        switti_wo_ddp,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse
        or isinstance(module, AdaLNSelfCrossAttn),
        device_id=dist.get_local_rank(),
        sharding_strategy=ShardingStrategy.NO_SHARD,  # Force no hybrid sharding
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
    )
    # build optimizer

    placeholder_tokens_content = [args.placeholder_token_content]
    placeholder_tokens_style = [args.placeholder_token_style]

    if args.num_vectors_content < 1 or args.num_vectors_style < 1:
        raise ValueError(
            f"--num_vectors_content and --num_vectors_style has to be larger or equal to 1, but is {args.num_vectors_content} and {args.num_vectors_style}"
        )

    # add dummy tokens for multi-vector
    additional_tokens_content = []
    for i in range(1, args.num_vectors_content):
        additional_tokens_content.append(f"{args.placeholder_token_content}_{i}")
    placeholder_tokens_content += additional_tokens_content

    additional_tokens_style = []
    for i in range(1, args.num_vectors_style):
        additional_tokens_style.append(f"{args.placeholder_token_style}_{i}")
    placeholder_tokens_style += additional_tokens_style

    placeholder_token = placeholder_tokens_content + placeholder_tokens_style
    print(f"placeholder_tokens: {placeholder_token}")

    num_added_tokens = pipe.text_encoder.tokenizer.add_tokens(placeholder_token)
    if num_added_tokens != (args.num_vectors_content + args.num_vectors_style):
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    num_added_tokens = pipe.text_encoder_2.tokenizer.add_tokens(placeholder_token)
    if num_added_tokens != (args.num_vectors_content + args.num_vectors_style):
        raise ValueError(
            f"The 2nd tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    if args.initializer_token_content:
        # Convert the initializer_token, placeholder_token to ids

        initializer_content_tokens = args.initializer_token_content.split("_")

        intializer_tokens_content_id = [
            pipe.text_encoder.tokenizer.encode(token, add_special_tokens=False)[0]
            for token in initializer_content_tokens
        ]
        intializer_tokens_content_id_2 = [
            pipe.text_encoder_2.tokenizer.encode(token, add_special_tokens=False)[0]
            for token in initializer_content_tokens
        ]

        if (
            len(intializer_tokens_content_id) < args.num_vectors_content
            and len(intializer_tokens_content_id) == 1
        ):
            # repeat the last token
            print(
                f"repeating the last token {intializer_tokens_content_id[0]} {args.num_vectors_content} times"
            )
            intializer_tokens_content_id = [
                intializer_tokens_content_id[0]
            ] * args.num_vectors_content
            intializer_tokens_content_id_2 = [
                intializer_tokens_content_id_2[0]
            ] * args.num_vectors_content

    if args.initializer_token_style:
        initializer_style_tokens = args.initializer_token_style.split("_")
        intializer_tokens_style_id = [
            pipe.text_encoder.tokenizer.encode(token, add_special_tokens=False)[0]
            for token in initializer_style_tokens
        ]
        intializer_tokens_style_id_2 = [
            pipe.text_encoder_2.tokenizer.encode(token, add_special_tokens=False)[0]
            for token in initializer_style_tokens
        ]

        if (
            len(intializer_tokens_style_id) < args.num_vectors_style
            and len(intializer_tokens_style_id) == 1
        ):
            # repeat the last token
            print(
                f"repeating the last token {intializer_tokens_style_id[0]} {args.num_vectors_style} times"
            )
            intializer_tokens_style_id = [intializer_tokens_style_id[0]] * args.num_vectors_style
            intializer_tokens_style_id_2 = [
                intializer_tokens_style_id_2[0]
            ] * args.num_vectors_style

    placeholder_token_content_ids = pipe.text_encoder.tokenizer.convert_tokens_to_ids(
        placeholder_tokens_content
    )
    placeholder_token_content_ids_2 = pipe.text_encoder_2.tokenizer.convert_tokens_to_ids(
        placeholder_tokens_content
    )

    placeholder_token_style_ids = pipe.text_encoder.tokenizer.convert_tokens_to_ids(
        placeholder_tokens_style
    )
    placeholder_token_style_ids_2 = pipe.text_encoder_2.tokenizer.convert_tokens_to_ids(
        placeholder_tokens_style
    )

    args.placeholder_token_content_ids = placeholder_token_content_ids
    args.placeholder_token_content_ids_2 = placeholder_token_content_ids_2
    args.placeholder_token_style_ids = placeholder_token_style_ids
    args.placeholder_token_style_ids_2 = placeholder_token_style_ids_2

    # save these ids to pipe
    pipe.kwargs_bonus["placeholder_token_content_ids"] = placeholder_token_content_ids
    pipe.kwargs_bonus["placeholder_token_content_ids_2"] = placeholder_token_content_ids_2
    pipe.kwargs_bonus["placeholder_token_style_ids"] = placeholder_token_style_ids
    pipe.kwargs_bonus["placeholder_token_style_ids_2"] = placeholder_token_style_ids_2

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    pipe.text_encoder.transformer.resize_token_embeddings(len(pipe.text_encoder.tokenizer))
    pipe.text_encoder_2.transformer.resize_token_embeddings(len(pipe.text_encoder_2.tokenizer))

    token_embeds = pipe.text_encoder.transformer.get_input_embeddings().weight.data
    token_embeds_2 = pipe.text_encoder_2.transformer.get_input_embeddings().weight.data
    if args.initializer_token_content:

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        with torch.no_grad():
            for idx, token_id in enumerate(placeholder_token_content_ids):
                token_embeds[token_id] = token_embeds[intializer_tokens_content_id[idx]].clone()

            for idx, token_id in enumerate(placeholder_token_content_ids_2):
                token_embeds_2[token_id] = token_embeds_2[
                    intializer_tokens_content_id_2[idx]
                ].clone()

    if args.initializer_token_style:
        with torch.no_grad():
            for idx, token_id in enumerate(placeholder_token_style_ids):
                token_embeds[token_id] = token_embeds[intializer_tokens_style_id[idx]].clone()

            for idx, token_id in enumerate(placeholder_token_style_ids_2):
                token_embeds_2[token_id] = token_embeds_2[intializer_tokens_style_id_2[idx]].clone()

    switti.requires_grad_(False)
    switti.virtual_token_embed.requires_grad_(True)  # Enable gradient for virtual tokens
    vae_local.requires_grad_(False)

    pipe.text_encoder.transformer.requires_grad_(True)
    pipe.text_encoder_2.transformer.requires_grad_(True)
    pipe.text_encoder.transformer.text_model.encoder.requires_grad_(False)
    pipe.text_encoder.transformer.text_model.final_layer_norm.requires_grad_(False)
    pipe.text_encoder.transformer.text_model.embeddings.position_embedding.requires_grad_(False)
    pipe.text_encoder_2.transformer.text_model.encoder.requires_grad_(False)
    pipe.text_encoder_2.transformer.text_model.final_layer_norm.requires_grad_(False)
    pipe.text_encoder_2.transformer.text_model.embeddings.position_embedding.requires_grad_(False)

    names_1, paras_1, para_groups_1 = filter_params(
        pipe.text_encoder.transformer,
        nowd_keys={
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
        select_params=["token_embedding"],
    )

    print(f"names_1: {names_1}")
    print(f"paras_1: {type(paras_1[0])} {paras_1}")
    print(f"para_groups_1: {para_groups_1}")

    names_2, paras_2, para_groups_2 = filter_params(
        pipe.text_encoder_2.transformer,
        nowd_keys={
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
        select_params=["token_embedding"],
    )

    print(f"names_2: {names_2}")
    print(f"paras_2: {paras_2}")
    print(f"para_groups_2: {para_groups_2}")

    # Add virtual token parameters
    virtual_token_param_group = {
        "params": [switti.virtual_token_embed.weight],
        "lr": args.tlr,
        "weight_decay": 0.0,
        "name": "virtual_token_embeddings",
    }

    # Combine all parameter groups
    names = names_1 + names_2 + ["virtual_token_embeddings"]
    paras = paras_1 + paras_2 + [switti.virtual_token_embed.weight]
    para_groups = para_groups_1 + para_groups_2 + [virtual_token_param_group]

    print(f"para_groups: {para_groups}")

    optimizer = torch.optim.AdamW(
        params=para_groups,
        lr=args.tlr,
        weight_decay=0.0,
        betas=(args.adam_beta1, args.adam_beta2),
        fused=args.afuse if not args.use_fsdp else False,
    )

    switti_optimizer = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=optimizer,
        names=names,
        paras=paras,
        grad_clip=args.tclip,
    )
    del names, paras, para_groups

    # build data
    print("[build PT data] ...\n")
    print(f"global bs={args.glb_batch_size}, local bs={args.batch_size}")

    place_holder_token_content_use = " ".join(
        pipe.text_encoder.tokenizer.convert_ids_to_tokens(placeholder_token_content_ids)
    )
    print(f"place_holder_token_content_use: {place_holder_token_content_use}")

    place_holder_token_style_use = " ".join(
        pipe.text_encoder.tokenizer.convert_ids_to_tokens(placeholder_token_style_ids)
    )
    print(f"place_holder_token_style_use: {place_holder_token_style_use}")

    args.placeholder_token_content_use = place_holder_token_content_use
    args.placeholder_token_style_use = place_holder_token_style_use
    args.placeholder_token_use = place_holder_token_content_use

    dataset_train = build_personalized_dataset_disentanglement(
        args.data_path,
        final_reso=args.data_load_reso,
        hflip=args.hflip,
        mid_reso=args.mid_reso,
        placeholder_token_content=args.placeholder_token_content_use,
        placeholder_token_style=args.placeholder_token_style_use,
        use_captions=args.use_captions,
        repeat=args.dataset_repeats,
    )

    ld_train = DataLoader(
        dataset=dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),  # worker_init_fn=worker_init_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            start_it=start_it,
        ),
    )
    del dataset_train

    # build trainer
    trainer = SwittiTrainerDisentanglement(
        dataloader=ld_train,
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        pipe=pipe,
        vae_local=vae_local,
        switti_wo_ddp=switti_wo_ddp,
        switti=switti,
        optimizer=switti_optimizer,
        label_smooth=args.ls,
        placeholder_token_ids=[placeholder_token_content_ids, placeholder_token_style_ids],
        placeholder_token_ids_2=[placeholder_token_content_ids_2, placeholder_token_style_ids_2],
        args=args,
    )
    torch.cuda.empty_cache()

    args.placeholder_token_ids = placeholder_token_content_ids + placeholder_token_style_ids
    args.placeholder_token_ids_2 = placeholder_token_content_ids_2 + placeholder_token_style_ids_2

    return (tb_lg, trainer, start_it)


def save_virtual_token_embed(switti: Switti, args: arg_util.Args, cur_iter: int) -> None:
    """
    Save virtual token embeddings to a file.

    Args:
        switti: Switti model containing virtual token embeddings
        args: Training arguments
        cur_iter: Current iteration number
    """
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    save_path = os.path.join(args.local_out_dir_path, f"virtual_token_embed-steps-{cur_iter}.pt")
    print(switti.virtual_token_embed.weight)
    torch.save(switti.virtual_token_embed.state_dict(), save_path)
    print(
        f"Saved virtual token embeddings {switti.virtual_token_embed.weight.shape} to {save_path}"
    )


def main_training() -> None:
    """
    Main training function that orchestrates the complete training process.

    Initializes distributed training, builds all components, and runs the
    training loop with proper logging and checkpointing.
    """
    torch.set_num_threads(32)
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    (tb_lg, trainer, start_it) = build_everything(args)
    dist.barrier()

    orig_embeds_params = (
        trainer.pipe.text_encoder.transformer.get_input_embeddings().weight.data.clone()
    )
    orig_embeds_params_2 = (
        trainer.pipe.text_encoder_2.transformer.get_input_embeddings().weight.data.clone()
    )

    save_training_config(args)

    for cur_iter in range(start_it, args.max_iters):
        tb_lg.set_step(cur_iter)

        # get current lr, wd
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.optimizer.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            cur_iter,
            args.wp,
            args.max_iters,
            wp0=args.wp0,
            wpe=args.wpe,
            wp_start_it=start_it,
        )
        args.cur_lr, args.cur_wd = max_tlr, max_twd

        # model forward-backward
        grad_norm, scale_log2 = trainer.train_step(
            g_it=cur_iter,
            tb_lg=tb_lg,
            orig_embeds_params=orig_embeds_params,
            orig_embeds_params_2=orig_embeds_params_2,
        )

        tb_lg.update(head="AR_opt_lr/lr_min", sche_tlr=min_tlr)
        tb_lg.update(head="AR_opt_lr/lr_max", sche_tlr=max_tlr)
        tb_lg.update(head="AR_opt_wd/wd_max", sche_twd=max_twd)
        tb_lg.update(head="AR_opt_wd/wd_min", sche_twd=min_twd)
        tb_lg.update(head="AR_opt_grad/fp16", scale_log2=scale_log2)
        if args.tclip > 0:
            tb_lg.update(head="AR_opt_grad/grad", grad_norm=grad_norm)
            tb_lg.update(head="AR_opt_grad/grad", grad_clip=args.tclip)

        if cur_iter % args.save_iters == 0 and cur_iter > start_it:

            # Save virtual token embeddings
            save_virtual_token_embed(trainer.pipe.switti, args, cur_iter)

            # save content
            args.placeholder_token = args.placeholder_token_content
            weight_name = f"learned_embeds-steps-{cur_iter}_content.safetensors"
            save_progress(
                trainer.pipe.text_encoder, args.placeholder_token_content_ids, args, weight_name
            )

            weight_name = f"learned_embeds_2-steps-{cur_iter}_content.safetensors"
            save_progress(
                trainer.pipe.text_encoder_2, args.placeholder_token_content_ids_2, args, weight_name
            )

            # save style
            args.placeholder_token = args.placeholder_token_style
            weight_name = f"learned_embeds-steps-{cur_iter}_style.safetensors"
            save_progress(
                trainer.pipe.text_encoder, args.placeholder_token_style_ids, args, weight_name
            )
            weight_name = f"learned_embeds_2-steps-{cur_iter}_style.safetensors"
            save_progress(
                trainer.pipe.text_encoder_2, args.placeholder_token_style_ids_2, args, weight_name
            )

    gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = "-", time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(time.time() - 60)
    )
    print(f"final args:\n\n{str(args)}")
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


class NullDDP(torch.nn.Module):
    """
    Null distributed data parallel wrapper for single-GPU training.

    This class provides a no-op wrapper when distributed training is not used,
    allowing the same code to work in both distributed and single-GPU scenarios.
    """

    def __init__(self, module: torch.nn.Module, *args, **kwargs) -> None:
        """
        Initialize the NullDDP wrapper.

        Args:
            module: The module to wrap
            *args: Additional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)
        """
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the wrapped module.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            Output from the wrapped module
        """
        return self.module(*args, **kwargs)


def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculate the memory size of a model in MB.

    Args:
        model: PyTorch model to measure

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def estimate_training_memory(
    model: torch.nn.Module, batch_size: int, optimizer_type: str = "adam"
) -> Dict[str, float]:
    """
    Estimate memory usage during training.

    Args:
        model: PyTorch model to estimate memory for
        batch_size: Training batch size
        optimizer_type: Type of optimizer used (default: "adam")

    Returns:
        Dictionary containing memory estimates in MB for different components
    """
    param_size = 0
    grad_size = 0

    # Parameter and gradient memory
    for param in model.parameters():
        if param.requires_grad:
            param_numel = param.nelement()
            param_size += param_numel * param.element_size()
            grad_size += (
                param_numel * param.element_size()
            )  # gradients typically same size as params

    # Optimizer states memory
    optimizer_memory = 0
    if optimizer_type.lower() == "adam":
        optimizer_memory = 2 * param_size  # Adam keeps 2 additional states (momentum and variance)

    # Rough activation memory estimation (this is approximate)
    # Typically 2-3x parameter size for transformer models
    activation_memory = 2.5 * param_size * batch_size

    # Backward pass typically needs about the same as forward
    backward_memory = activation_memory

    total_memory = {
        "parameters": param_size / 1024**2,  # MB
        "gradients": grad_size / 1024**2,
        "optimizer_states": optimizer_memory / 1024**2,
        "activation_forward": activation_memory / 1024**2,
        "activation_backward": backward_memory / 1024**2,
        "total": (param_size + grad_size + optimizer_memory + activation_memory + backward_memory)
        / 1024**2,
    }

    return total_memory


if __name__ == "__main__":
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
