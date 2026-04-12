
# Code adapted from:
# https://github.com/yandex-research/switti/blob/master/models/switti.py

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.embeddings import GaussianFourierProjection
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_switti import AdaLNBeforeHead, AdaLNSelfCrossAttn
from models.rope import compute_axial_cis


def get_crop_condition(heights: list, widths: list, base_size=512):
    if type(heights[0]) == type(widths[0]) == str:
        heights = [int(h) for h in heights]
        widths = [int(w) for w in widths]
    h = torch.tensor(heights, dtype=torch.int).unsqueeze(1)
    w = torch.tensor(widths, dtype=torch.int).unsqueeze(1)
    hw = torch.cat([h, w], dim=1)

    ratio = base_size / hw.min(-1)[0]
    orig_size = (hw * ratio[:, None]).to(torch.int)
    crop_coords = ((orig_size - base_size) // 2).clamp(min=0)
    crop_cond = torch.cat([orig_size, crop_coords], dim=1)

    return crop_cond


class Switti(nn.Module):
    def __init__(
        self,
        Cvae=32,
        V=4096,
        rope=True,
        rope_theta=10000,
        rope_size=128,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        attn_l2_norm=True,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        fused_if_available=True,
        use_swiglu_ffn=True,
        use_ar=False,
        use_crop_cond=True,
        prompt_config=None,
    ):
        super().__init__()

        # assert mode selected_single_scales then selected_scales must be provided
        # assert mode == "selected_single_scales" and selected_scales is not None

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )
        self.Cvae, self.V = Cvae, V

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.rope = rope

        # Fix for the reduce() error - create a list of the patch dimensions
        patch_dimensions = [pn**2 for pn in self.patch_nums]

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. text embedding
        self.pooled_embed_size = 1280
        self.context_dim = 1280 + 768
        self.text_pooler = nn.Linear(self.pooled_embed_size, self.D)

        init_std = math.sqrt(1 / self.C / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        #  prompt config
        if prompt_config is None:
            prompt_config = {
                "selected_scales": [10],
                "num_virtual_tokens_per_scale": 1,
                "add_mode": "selected_single_scales",
                "deep": {
                    "is_deep": False,
                },
                "init_zeros": False,
            }

        print("Prompt config: ", prompt_config)

        self.selected_scales = prompt_config["selected_scales"]
        self.num_virtual_tokens_per_scale = prompt_config["num_virtual_tokens_per_scale"]
        self.add_mode = prompt_config["add_mode"]
        self.deep = prompt_config["deep"][0]["is_deep"]
        self.mode_layer = prompt_config["deep"][1]["mode_layer"]
        self.init_zeros = prompt_config["init_zeros"]

        assert self.add_mode in ["selected_single_scales", "group_scales"]
        assert self.selected_scales is not None
        # if mode is group_scales, len(selected_scales) must be 2
        if self.deep:
            assert self.mode_layer in ["all", "half"]
        if self.add_mode == "group_scales":
            assert len(self.selected_scales) == 2  # [0, 3]

        print(f"deep: {self.deep}, mode_layer: {self.mode_layer}")

        # 3. position embedding

        self.non_virtual_indices = []
        cur_idx = 0
        for i in range(len(self.patch_nums)):
            self.non_virtual_indices.append((cur_idx, cur_idx + self.patch_nums[i] ** 2))
            cur_idx += self.patch_nums[i] ** 2

        if not self.rope:
            # absolute position embedding
            pos_1LC = []
            for i, pn in enumerate(self.patch_nums):
                pe = torch.empty(1, pn * pn, self.C)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
            pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
            assert tuple(pos_1LC.shape) == (1, self.L, self.C)
            self.pos_1LC = nn.Parameter(pos_1LC)
            self.freqs_cis = None
        else:
            # RoPE position embedding
            assert (
                self.C // self.num_heads
            ) % 4 == 0, "2d rope needs head dim to be divisible by 4"
            patch_nums_m1 = tuple(pn - 1 if pn > 1 else 1 for pn in self.patch_nums)
            self.compute_cis = partial(compute_axial_cis, dim=self.C // self.num_heads)
            freqs_cis = []
            for i, pn in enumerate(self.patch_nums):
                norm_coeff = rope_size / patch_nums_m1[i]
                cur_freqs = self.compute_cis(
                    end_x=pn, end_y=pn, theta=rope_theta, norm_coeff=norm_coeff
                )
                freqs_cis.append(cur_freqs[None, ...])
            self.freqs_cis = torch.cat(freqs_cis, dim=1)  # 1, L, C // 2 -- complex

        self.added_virtual_tokens = [
            self.num_virtual_tokens_per_scale if i in self.selected_scales else 0
            for i in range(len(self.patch_nums))
        ]

        # level embedding (similar to GPT's segment embedding,
        # used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.drop_path_rate = drop_path_rate
        # stochastic depth decay rule (linearly increasing)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([])
        for block_idx in range(depth):
            self.blocks.append(
                AdaLNSelfCrossAttn(
                    cond_dim=self.D,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    qk_norm=attn_l2_norm,
                    context_dim=self.context_dim,
                    use_swiglu_ffn=use_swiglu_ffn,
                    norm_eps=norm_eps,
                    use_crop_cond=use_crop_cond,
                )
            )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f"\n[constructor]  ==== fused_if_available={fused_if_available} "
            f"(fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, "
            f"fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n"
            f"    [Switti config ] embed_dim={embed_dim}, num_heads={num_heads}, "
            f"depth={depth}, mlp_ratio={mlp_ratio}\n"
            f"    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, "
            f"drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})",
            end="\n\n",
            flush=True,
        )

        # Prepare crop condition embedder
        self.use_crop_cond = use_crop_cond
        if use_crop_cond:
            # crop condition is repredsented with 4 int values. each is embeded to self.D // 4 dim
            assert self.D % 8 == 0
            self.crop_embed = GaussianFourierProjection(
                self.D // 2 // 4, set_W_to_weight=False, log=False, flip_sin_to_cos=False
            )
            self.crop_proj = nn.Linear(self.D, self.D)

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        self.use_ar = use_ar
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)

        # d_virtual: torch.Tensor = torch.cat(
        #     [torch.full((pn * pn + num_virtual_tokens_per_scale,), i) for i, pn in enumerate(self.patch_nums)]
        # ).view(1, self.L + num_virtual_tokens_per_scale * len(self.patch_nums), 1)

        d_virtual: torch.Tensor = torch.cat(
            [
                torch.full((pn * pn + self.added_virtual_tokens[i],), i)
                for i, pn in enumerate(self.patch_nums)
            ]
        ).view(1, self.L + sum(self.added_virtual_tokens), 1)

        dT = d.transpose(1, 2)  # dT: 11L
        dT_virtual = d_virtual.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)

        if self.use_ar:
            attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf)
            attn_bias_for_masking_virtual = torch.where(d >= dT_virtual, 0.0, -torch.inf)
        else:
            attn_bias_for_masking = torch.where(d == dT, 0.0, -torch.inf)
            attn_bias_for_masking_virtual = torch.where(d == dT_virtual, 0.0, -torch.inf)

        if self.add_mode == "group_scales":

            cur_idx = 0
            actual_cur_idx = 0
            for i, (pn, num_vt) in enumerate(zip(self.patch_nums, self.added_virtual_tokens)):

                if num_vt > 0:
                    virtual_token_start = cur_idx
                    virtual_token_end = cur_idx + num_vt

                    if i == self.selected_scales[1]:
                        attn_bias_for_masking_virtual[
                            :, :actual_cur_idx, : self.num_virtual_tokens_per_scale
                        ] = 0.0
                        attn_bias_for_masking_virtual[
                            :, actual_cur_idx:, virtual_token_start:virtual_token_end
                        ] = 0.0

                cur_idx += num_vt + pn * pn
                actual_cur_idx += pn * pn

        attn_bias_for_masking = attn_bias_for_masking.reshape(1, 1, self.L, self.L)
        # attn_bias_for_masking_virtual = attn_bias_for_masking_virtual.reshape(1, 1, self.L, self.L + num_virtual_tokens_per_scale * len(self.patch_nums))
        attn_bias_for_masking_virtual = attn_bias_for_masking_virtual.reshape(
            1, 1, self.L, self.L + sum(self.added_virtual_tokens)
        )

        self.register_buffer("attn_bias_for_masking", attn_bias_for_masking.contiguous())
        self.register_buffer(
            "attn_bias_for_masking_virtual", attn_bias_for_masking_virtual.contiguous()
        )

        # 6. classifier head
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # By default disable gradient checkpointing
        self.use_gradient_checkpointing = False

        print(f"num_virtual_tokens_per_scale: {self.num_virtual_tokens_per_scale}")
        self.num_virtual_tokens_per_scale = self.num_virtual_tokens_per_scale

        if self.deep:
            if self.mode_layer == "all":
                self.num_insert_blocks = len(self.blocks)
            elif self.mode_layer == "half":
                self.num_insert_blocks = len(self.blocks) // 2
        else:
            self.num_insert_blocks = 1

        self.virtual_token_embed = nn.Embedding(
            len(self.patch_nums) * self.num_virtual_tokens_per_scale,
            self.num_insert_blocks * self.C * 2,
        )
        if not self.init_zeros:
            # Fix the reduce calculation
            val = math.sqrt(6.0 / float(3 * sum(patch_dimensions) + self.C))

            # xavier_uniform initialization
            # nn.init.uniform_(self.virtual_token_embed.weight[:, :self.C], -val, val)
            # nn.init.uniform_(self.virtual_token_embed.weight[:, self.C:], -val, val)

            for i in range(self.num_insert_blocks):
                start_idx = i * self.C * 2
                end_idx = (i + 1) * self.C * 2
                nn.init.uniform_(
                    self.virtual_token_embed.weight[:, start_idx : start_idx + self.C], -val, val
                )
                nn.init.uniform_(
                    self.virtual_token_embed.weight[:, start_idx + self.C : end_idx], -val, val
                )

            # nn.init.xavier_uniform_(self.virtual_token_embed.weight)
        else:
            # Initialize weights to zero
            nn.init.zeros_(self.virtual_token_embed.weight)

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h, cond_BD))

    def forward_content_style(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_attn_bias: torch.Tensor,
        prompt_content_embeds: torch.Tensor,
        prompt_content_attn_bias: torch.Tensor,
        prompt_style_embeds: torch.Tensor,
        prompt_style_attn_bias: torch.Tensor,
        batch_height: list[int] | None = None,
        batch_width: list[int] | None = None,
        is_train_style: int = 1,
    ):
        # Insert virtual tokens before each scale
        virtual_tokens = []
        for i, pn in enumerate(self.patch_nums):
            virtual_token_idx = torch.full(
                (1, self.num_virtual_tokens_per_scale),
                i * self.num_virtual_tokens_per_scale,
                dtype=torch.long,
                device=x_BLCv_wo_first_l.device,
            )
            virtual_tokens.append(self.virtual_token_embed(virtual_token_idx))

        # Use the original input as the query
        # query_input = x_BLCv_wo_first_l

        bg, ed = 0, self.L
        B = x_BLCv_wo_first_l.shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            pooled_prompt_embeds = self.text_pooler(pooled_prompt_embeds)

            sos = cond_BD = pooled_prompt_embeds
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(
                B, self.first_l, -1
            )

            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))  # lvl: BLC;  pos: 1LC
            if not self.rope:
                x_BLC += self.pos_1LC[:, :ed]

            # query_input = x_BLC.clone()
            # add virtual tokens at this stage
            # Concatenate virtual tokens with the input
            cur_idx = 0
            # list_x_BLC = []
            non_virtual_indices = []
            for i in range(len(self.patch_nums)):
                # Expand virtual tokens to match the batch size
                # expanded_virtual_tokens = virtual_tokens[i].expand(x_BLCv_wo_first_l.size(0), -1, -1)
                # list_x_BLC.append(torch.cat([expanded_virtual_tokens, x_BLC[:, cur_idx:cur_idx + self.patch_nums[i]**2].clone()], dim=1))
                pre_added_virtual_tokens = sum(self.added_virtual_tokens[:i])
                non_virtual_indices.append(
                    torch.arange(
                        cur_idx + pre_added_virtual_tokens,
                        cur_idx + pre_added_virtual_tokens + self.patch_nums[i] ** 2,
                    )
                )
                cur_idx += self.patch_nums[i] ** 2
            # x_BLC = torch.cat(list_x_BLC, dim=1)
            non_virtual_indices = torch.cat(non_virtual_indices)
            non_virtual_indices = non_virtual_indices.long()

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        attn_bias_virtual = self.attn_bias_for_masking_virtual[
            :, :, :ed, : ed + sum(self.added_virtual_tokens)
        ]

        if self.use_crop_cond:
            crop_coords = get_crop_condition(batch_height, batch_width).to(cond_BD.device)
            crop_embed = self.crop_embed(crop_coords.view(-1)).reshape(B, self.D)
            crop_cond = self.crop_proj(crop_embed)
        else:
            crop_cond = None

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD = cond_BD.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # non_virtual_indices = []
        # cur_idx = 0
        # for i in range(len(virtual_tokens)):
        #     non_virtual_indices.append((cur_idx, cur_idx + self.patch_nums[i]**2))
        #     cur_idx += self.patch_nums[i]**2

        # always insert all blocks [MAGIC]
        # insert_idx_block = range(len(self.blocks))
        insert_idx_block = range(self.num_insert_blocks)

        for idx_block, block in enumerate(self.blocks):

            query_input = x_BLC.clone()
            list_x_BLC = []

            # cur_idx = 0
            # for i in range(len(self.patch_nums)):
            #     # Expand virtual tokens to match the batch size
            #     expanded_virtual_tokens = virtual_tokens[i].expand(x_BLCv_wo_first_l.size(0), -1, -1)
            #     list_x_BLC.append(torch.cat([expanded_virtual_tokens, x_BLC[:, cur_idx:cur_idx + self.patch_nums[i]**2].clone()], dim=1))
            #     cur_idx += self.patch_nums[i]**2

            # x_BLC = torch.cat(list_x_BLC, dim=1)

            expanded_virtual_tokens_list = []
            for i in range(len(virtual_tokens)):

                if self.deep:
                    start_idx = idx_block * self.C * 2
                    end_idx = (idx_block + 1) * self.C * 2
                else:
                    start_idx = 0
                    end_idx = self.C * 2

                expanded_virtual_tokens = virtual_tokens[i].expand(
                    x_BLCv_wo_first_l.size(0), -1, -1
                )[:, :, start_idx:end_idx]
                expanded_virtual_tokens_list.append(expanded_virtual_tokens)

            if self.use_gradient_checkpointing:
                x_BLC = torch.utils.checkpoint.checkpoint(
                    block,
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=attn_bias_virtual if idx_block in insert_idx_block else attn_bias,
                    context=prompt_style_embeds if is_train_style == 1 else prompt_content_embeds,
                    freqs_cis=self.freqs_cis,
                    context_attn_bias=(
                        prompt_style_attn_bias if is_train_style == 1 else prompt_content_attn_bias
                    ),
                    crop_cond=crop_cond,
                    virtual_kv=(
                        expanded_virtual_tokens_list if idx_block in insert_idx_block else None
                    ),
                    non_virtual_indices=self.non_virtual_indices,
                    selected_scales=self.selected_scales,
                    use_reentrant=False,
                )
            else:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=attn_bias_virtual if idx_block in insert_idx_block else attn_bias,
                    context=prompt_style_embeds if is_train_style == 1 else prompt_content_embeds,
                    freqs_cis=self.freqs_cis,
                    context_attn_bias=(
                        prompt_style_attn_bias if is_train_style == 1 else prompt_content_attn_bias
                    ),
                    crop_cond=crop_cond,
                    virtual_kv=(
                        expanded_virtual_tokens_list if idx_block in insert_idx_block else None
                    ),
                    non_virtual_indices=self.non_virtual_indices,
                    selected_scales=self.selected_scales,
                )

        with torch.amp.autocast("cuda", enabled=not self.training):
            x_BLC = self.get_logits(x_BLC, cond_BD.float())

        return x_BLC  # logits BLV, V is vocab_size

    def forward(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_attn_bias: torch.Tensor,
        batch_height: list[int] | None = None,
        batch_width: list[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:  # returns logits_BLV
        """
            :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
            :param prompt_embeds (B, context_len, self.context_dim):
            text features from pipe.text_encoder and pipe.text_encoder_2,
            concatenated along dim=-1, padded to longest along dim=1
            :param pooled_prompt_embeds (B, self.pooled_embed_size):
            pooled text features from pipe.text_encoder_2
            :param prompt_attn_bias (B, context_len):
            boolean mask to specify which tokens are not padding
            :param batch_height (B,): original height of images in a batch.
        :param batch_width (B,): original width of images in a batch.
            Only used when self.use_crop_cond = True
            :return: logits BLV, V is vocab_size
        """
        if (
            kwargs.get("prompt_content_embeds") is not None
            and kwargs.get("prompt_style_embeds") is not None
        ):
            prompt_content_embeds = kwargs.get("prompt_content_embeds")
            prompt_content_attn_bias = kwargs.get("prompt_content_attn_bias")
            prompt_style_embeds = kwargs.get("prompt_style_embeds")
            prompt_style_attn_bias = kwargs.get("prompt_style_attn_bias")
            is_train_style = kwargs.get("is_train_style")
            return self.forward_content_style(
                x_BLCv_wo_first_l,
                prompt_embeds,
                pooled_prompt_embeds,
                prompt_attn_bias,
                prompt_content_embeds,
                prompt_content_attn_bias,
                prompt_style_embeds,
                prompt_style_attn_bias,
                batch_height,
                batch_width,
                is_train_style,
            )

        # Insert virtual tokens before each scale
        virtual_tokens = []
        for i, pn in enumerate(self.patch_nums):
            virtual_token_idx = torch.full(
                (x_BLCv_wo_first_l.size(0), self.num_virtual_tokens_per_scale),
                i * self.num_virtual_tokens_per_scale,
                dtype=torch.long,
                device=x_BLCv_wo_first_l.device,
            )
            virtual_tokens.append(self.virtual_token_embed(virtual_token_idx))

        # Concatenate virtual tokens with the input
        x_BLC = torch.cat(
            [
                [vt, x_BLCv_wo_first_l[:, i * pn**2 : (i + 1) * pn**2]]
                for i, vt in enumerate(virtual_tokens)
            ],
            dim=1,
        )

        bg, ed = 0, self.L
        B = x_BLCv_wo_first_l.shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            pooled_prompt_embeds = self.text_pooler(pooled_prompt_embeds)

            sos = cond_BD = pooled_prompt_embeds
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(
                B, self.first_l, -1
            )

            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))  # lvl: BLC;  pos: 1LC
            if not self.rope:
                x_BLC += self.pos_1LC[:, :ed]
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]

        if self.use_crop_cond:
            crop_coords = get_crop_condition(batch_height, batch_width).to(cond_BD.device)
            crop_embed = self.crop_embed(crop_coords.view(-1)).reshape(B, self.D)
            crop_cond = self.crop_proj(crop_embed)
        else:
            crop_cond = None

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD = cond_BD.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_BLC = torch.utils.checkpoint.checkpoint(
                    block,
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=attn_bias,
                    context=prompt_embeds,
                    freqs_cis=self.freqs_cis,
                    context_attn_bias=prompt_attn_bias,
                    crop_cond=crop_cond,
                    use_reentrant=False,
                )
            else:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=attn_bias,
                    context=prompt_embeds,
                    freqs_cis=self.freqs_cis,
                    context_attn_bias=prompt_attn_bias,
                    crop_cond=crop_cond,
                )

        with torch.amp.autocast("cuda", enabled=not self.training):
            x_BLC = self.get_logits(x_BLC, cond_BD.float())

        return x_BLC  # logits BLV, V is vocab_size

    def init_weights(
        self,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=0.02,
    ):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if (
                hasattr(self.head_nm.ada_lin[-1], "bias")
                and self.head_nm.ada_lin[-1].bias is not None
            ):
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block in self.blocks:
            block.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            block.cross_attn.proj.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(block.ffn, "fc2"):
                block.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))

            if hasattr(block, "ada_lin"):
                block.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                block.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                if hasattr(block.ada_lin[-1], "bias") and block.ada_lin[-1].bias is not None:
                    block.ada_lin[-1].bias.data.zero_()
            elif hasattr(block, "ada_gss"):
                block.ada_gss.data[:, :, 2:].mul_(init_adaln)
                block.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

        # Initialize virtual token embeddings to zero
        if hasattr(self, "virtual_token_embed"):
            nn.init.zeros_(self.virtual_token_embed.weight)

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"


class SwittiHF(Switti, PyTorchModelHubMixin):
    # tags=["image-generation"]):
    def __init__(
        self,
        depth=30,
        rope=True,
        rope_theta=10000,
        rope_size=128,
        use_swiglu_ffn=True,
        use_ar=False,
        use_crop_cond=True,
        prompt_config=None,
    ):
        heads = depth
        width = depth * 64

        print("Init param", use_crop_cond, type(use_crop_cond))

        super().__init__(
            depth=depth,
            embed_dim=width,
            num_heads=heads,
            patch_nums=(1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
            rope=rope,
            rope_theta=rope_theta,
            rope_size=rope_size,
            use_swiglu_ffn=use_swiglu_ffn,
            use_ar=use_ar,
            use_crop_cond=use_crop_cond,
            prompt_config=prompt_config,
        )
