# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import nn


class Qwen2RotaryPositionalEmbeddings(nn.Module):
    """
    RoPE Embeddings used in the Qwen2 model.
    Ref: https://huggingface.co/Qwen/Qwen2-7B-Instruct

    This class is not numerically equivalent to the RoPE Embedding module
    used by Llama2 and Llama3.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim`` // ``num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (float): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2)[:(self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len,
                               dtype=self.theta.dtype,
                               device=self.theta.device)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # We cache the cos and sin embeddings instead of the IDs. This helps
        # ensure we have correct behavior when training with bf16
        # Size: [max_seq_len, (dim * 2)]
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def _maybe_grow_cache(self, needed_seq_len: int) -> None:
        """Grow cached RoPE table if a requested position exceeds cache."""
        if needed_seq_len <= self.cache.shape[0]:
            return
        # Grow geometrically to avoid frequent reallocations on long trajectories.
        new_len = max(needed_seq_len, self.cache.shape[0] * 2)
        self.max_seq_len = int(new_len)
        self.build_rope_cache(self.max_seq_len)

    def forward(self,
                x: torch.Tensor,
                input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)
        head_dim = x.size(-1)
        if head_dim != self.dim:
            raise ValueError(
                f"RoPE dim mismatch: got head_dim={head_dim}, expected {self.dim}"
            )
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

        if input_pos is None:
            needed_seq_len = seq_len
        else:
            input_pos = input_pos.to(torch.long)
            needed_seq_len = int(torch.max(input_pos).item()) + 1
        self._maybe_grow_cache(needed_seq_len)

        # extract the values based on whether input_pos is set or not. When
        # input_pos is provided, we're in inference mode
        if input_pos is None:
            rope_cache = self.cache[:seq_len]  # (s, 2*h_d)
            rope_cache = rope_cache.unsqueeze(0).unsqueeze(2)  # (1,s,1,2*h_d)
        else:
            if input_pos.dim() == 1:
                # (s,) -> (1,s,1,2*h_d)
                rope_cache = self.cache[input_pos].unsqueeze(0).unsqueeze(2)
            else:
                # (b,s) -> (b,s,1,2*h_d)
                rope_cache = self.cache[input_pos].unsqueeze(2)

        # [b, s, 1, h_d]
        cos = rope_cache[..., :head_dim].to(x.dtype)
        sin = rope_cache[..., head_dim:].to(x.dtype)

        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)

        # cos: [b, s, 1, h_d]
        # x: [b, s, n_h, h_d]
        x_out = (x * cos) + (rotated * sin)
        return x_out.type_as(x)
