# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

from ..modules.transformer import FFN


class ContactHead(nn.Module):
    """
    Predict per-vertex contact states from contact query tokens.

    Takes all contact tokens (corresponding to the first 21 MHR70 keypoints:
    body joints + toes/heels) and predicts binary contact for each of the
    18439 MHR mesh vertices.

    Architecture:
        1. Mean-pool the num_contact_tokens tokens: [B, num_contact_tokens, C] -> [B, C]
        2. Deep MLP: [B, C] -> [B, num_vertices]
    """

    NUM_VERTICES = 18439

    def __init__(
        self,
        input_dim: int,
        num_contact_tokens: int = 21,
        num_vertices: int = 18439,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
    ):
        super().__init__()

        self.num_contact_tokens = num_contact_tokens
        self.num_vertices = num_vertices

        # Pool all contact tokens -> project to per-vertex logits.
        # The FFN operates on [B, 1, C] (token dim 1 after mean-pooling) and
        # outputs [B, 1, num_vertices], squeezed to [B, num_vertices].
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=num_vertices,
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: contact tokens  [B, num_contact_tokens, C]

        Returns:
            contact_logits: [B, num_vertices]  (un-sigmoid-ed)
        """
        batch_size, num_tokens, C = x.shape
        assert num_tokens == self.num_contact_tokens, (
            f"Expected {self.num_contact_tokens} contact tokens, got {num_tokens}"
        )

        # Mean pool over contact tokens: [B, num_tokens, C] -> [B, 1, C]
        x_pooled = x.mean(dim=1, keepdim=True)

        # [B, 1, C] -> [B, 1, num_vertices] -> [B, num_vertices]
        contact_logits = self.proj(x_pooled).squeeze(1)

        return contact_logits
