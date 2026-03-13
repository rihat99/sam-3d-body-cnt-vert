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
        1. Attention-weighted pool: a learnable query attends over all
           num_contact_tokens tokens -> [B, 1, C]
        2. Deep MLP: [B, C] -> [B, num_vertices]

    Compared to simple mean-pooling, attention-weighted pooling lets the model
    learn to up-weight tokens that are more informative for contact prediction
    rather than treating all joint tokens equally.
    """

    NUM_VERTICES = 18439

    def __init__(
        self,
        input_dim: int,
        num_contact_tokens: int = 21,
        num_vertices: int = 18439,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        pool_num_heads: int = 8,
    ):
        super().__init__()

        self.num_contact_tokens = num_contact_tokens
        self.num_vertices = num_vertices

        # Learnable query that attends over the contact tokens to produce a
        # single pooled representation.  Shape: [1, 1, input_dim].
        self.pool_query = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)

        # Multi-head attention: pool_query (Q) over contact tokens (K, V).
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=pool_num_heads,
            batch_first=True,
        )

        # Project pooled representation to per-vertex logits.
        # The FFN operates on [B, 1, C] and outputs [B, 1, num_vertices],
        # squeezed to [B, num_vertices].
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
        batch_size = x.shape[0]

        # Attention-weighted pool: [B, 1, C] x [B, num_tokens, C] -> [B, 1, C]
        query = self.pool_query.expand(batch_size, -1, -1)
        x_pooled, _ = self.pool_attn(query, x, x)  # [B, 1, C]

        # [B, 1, C] -> [B, 1, num_vertices] -> [B, num_vertices]
        contact_logits = self.proj(x_pooled).squeeze(1)

        return contact_logits
