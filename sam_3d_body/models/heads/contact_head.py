# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional

import torch
import torch.nn as nn

from ..modules.transformer import FFN


class ContactHead(nn.Module):
    """
    Predict contact states for body parts (feet and hands).
    
    Outputs binary classification (contact/no-contact) for:
    - Left foot (index 0)
    - Right foot (index 1)
    - Left hand (index 2)
    - Right hand (index 3)
    """

    def __init__(
        self,
        input_dim: int,
        num_contacts: int = 4,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
    ):
        super().__init__()

        self.num_contacts = num_contacts
        
        # Each contact token gets projected to a binary classification
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=1,  # Binary classification per token
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Args:
            x: contact tokens with shape [B, num_contacts, C]
               where C is the decoder dimension (e.g., 1024)
               
        Returns:
            contact_logits: [B, num_contacts] - logits for contact prediction
                - Index 0: Left foot
                - Index 1: Right foot  
                - Index 2: Left hand
                - Index 3: Right hand
        """
        batch_size, num_tokens, _ = x.shape
        assert num_tokens == self.num_contacts, \
            f"Expected {self.num_contacts} contact tokens, got {num_tokens}"
        
        # Project each token to a single logit
        # x: [B, num_contacts, C] -> [B, num_contacts, 1] -> [B, num_contacts]
        contact_logits = self.proj(x).squeeze(-1)
        
        return contact_logits
