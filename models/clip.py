
# Code adapted from:
# https://github.com/yandex-research/switti/blob/master/models/clip.py

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        dtype=torch.float32,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device, dtype)
        self.device = device
        self.hidden_size = self.transformer.config.hidden_size
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        pass
        # self.transformer = self.transformer.eval()
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, text, **kwargs):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        outputs = self.transformer(**batch_encoding)

        attn_bias = batch_encoding["attention_mask"].to(outputs["last_hidden_state"].dtype)
        attn_bias[attn_bias == 0] = -float("inf")
        attn_bias[attn_bias == 1] = 0.0
        outputs["attn_bias"] = attn_bias

        if (
            kwargs.get("proj_matrix") is not None
            and kwargs.get("proj_matrix_2") is not None
            and kwargs.get("placeholder_token_style_ids") is not None
            and kwargs.get("placeholder_token_style_ids_2") is not None
        ):
            proj_matrix = kwargs.get("proj_matrix")
            proj_matrix_2 = kwargs.get("proj_matrix_2")
            placeholder_token_style_ids = kwargs.get("placeholder_token_style_ids")
            placeholder_token_style_ids_2 = kwargs.get("placeholder_token_style_ids_2")

            proj_matrix = proj_matrix.to(
                batch_encoding["input_ids"].device, dtype=outputs["last_hidden_state"].dtype
            )
            proj_matrix_2 = proj_matrix_2.to(
                batch_encoding["input_ids"].device, dtype=outputs["last_hidden_state"].dtype
            )
            placeholder_token_style_ids = torch.tensor(placeholder_token_style_ids).to(
                batch_encoding["input_ids"].device, dtype=batch_encoding["input_ids"].dtype
            )
            placeholder_token_style_ids_2 = torch.tensor(placeholder_token_style_ids_2).to(
                batch_encoding["input_ids"].device, dtype=batch_encoding["input_ids"].dtype
            )

            if self.hidden_size == 768:
                mask_found_token = (
                    batch_encoding["input_ids"].unsqueeze(-1) == placeholder_token_style_ids
                ).any(dim=-1)
                positions = mask_found_token.nonzero(as_tuple=False)
                matched_embeddings = outputs["last_hidden_state"][
                    positions[:, 0], positions[:, 1], :
                ]
                matched_embeddings = matched_embeddings - matched_embeddings @ proj_matrix

                # matched_embeddings = matched_embeddings @ proj_matrix

                outputs["last_hidden_state"][
                    positions[:, 0], positions[:, 1], :
                ] = matched_embeddings

            else:
                mask_found_token_2 = (
                    batch_encoding["input_ids"].unsqueeze(-1) == placeholder_token_style_ids_2
                ).any(dim=-1)
                positions_2 = mask_found_token_2.nonzero(as_tuple=False)
                matched_embeddings_2 = outputs["last_hidden_state"][
                    positions_2[:, 0], positions_2[:, 1], :
                ]
                matched_embeddings_2 = matched_embeddings_2 - matched_embeddings_2 @ proj_matrix_2

                # matched_embeddings_2 = matched_embeddings_2 @ proj_matrix_2

                outputs["last_hidden_state"][
                    positions_2[:, 0], positions_2[:, 1], :
                ] = matched_embeddings_2

        return outputs

    # @torch.no_grad()
    def encode(self, text, **kwargs):
        return self(text, **kwargs)
