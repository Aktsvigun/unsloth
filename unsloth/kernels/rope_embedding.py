from typing import Optional, Any

# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch
from .utils import calculate_settings, torch_cuda_device

ROPE_GROUP_SIZE: int = 4


def _rope_embedding(
    Q: torch.Tensor,
    Q_row_stride: int,
    cos: torch.Tensor,
    cos_row_stride: int,
    sin: torch.Tensor,
    sin_row_stride: int,
    seqlen: int,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Calculates the RoPE Embedding quickly
    RoPE is Q * cos + rotate_half(Q) * sin
    See our blog post for more info
    """
    ROPE_GROUP_SIZE = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(
        sin
        + (row_position % seqlen) * sin_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask=mask,
        other=0,
    )
    cos1 = tl.load(
        cos
        + (row_position % seqlen) * cos_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask=mask,
        other=0,
    )

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1
    pass

    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # 10% Faster kernel from [HuyNguyen-hust](https://github.com/unslothai/unsloth/pull/238)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = (
            row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
        )

        # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
    pass


pass
_rope_embedding = triton.jit(_rope_embedding)
_rope_embedding = triton.heuristics(
    {
        "BACKWARD_PASS": lambda args: bool(args["BACKWARD_PASS"]),
    }
)(_rope_embedding)


class Fast_RoPE_Embedding(torch.autograd.Function):
    """
    Fast RoPE embedding implementation using Triton for efficient computation.

    This class implements the RoPE (Rotary Positional Embedding) operation with a highly optimized Triton kernel.
    It's designed to be faster than the standard implementation while maintaining numerical accuracy.

    RoPE is applied as: Q * cos + rotate_half(Q) * sin

    See the original paper for more details about Rotary Positional Embeddings.
    """

    @staticmethod
    def forward(
        ctx, Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Fast RoPE embedding.

        Args:
                ctx: Context object for saving values for backward pass
                Q (`torch.Tensor`): Query tensor of shape (batch, seq_len, n_heads, head_dim)
                cos (`torch.Tensor`): Cosine values tensor
                sin (`torch.Tensor`): Sine values tensor

        Returns:
                `torch.Tensor`: Tensor with RoPE embedding applied, same shape as input Q
        """
        cos, sin = cos.squeeze(), sin.squeeze()
        batch: int
        seq_len: int
        n_heads: int
        head_dim: int
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch * seq_len, n_heads * head_dim)
        n_rows: int
        n_cols: int
        n_rows, n_cols = Q.shape
        assert seq_len <= cos.shape[0]

        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)  # (head_dim//2)

        # group_size = 4 # 4 or 8, too large group_size can hurt performance.
        div: int
        mod: int
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups: int = div + (mod != 0)

        with torch_cuda_device(Q.device):
            _rope_embedding[
                (
                    n_rows,
                    n_groups,
                )
            ](
                Q,
                Q.stride(0),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS=False,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        return Q.view(batch, seq_len, n_heads, head_dim)

    pass

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """
        Backward pass of the Fast RoPE embedding.

        Args:
                ctx: Context object containing saved values from forward pass
                dY (`torch.Tensor`): Gradient tensor

        Returns:
                tuple[torch.Tensor, None, None]: Gradient with respect to input Q, with None values for cos and sin (no gradients computed for these)
        """
        batch: int
        seq_len: int
        n_heads: int
        head_dim: int
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch * seq_len, n_heads * head_dim)
        # Must be reshape not view
        n_rows: int
        n_cols: int
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin

        with torch_cuda_device(dY.device):
            _rope_embedding[
                (
                    n_rows,
                    ctx.n_groups,
                )
            ](
                dY,
                dY.stride(0),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS=True,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )
        dY = dY.view(batch, seq_len, n_heads, head_dim)
        return (
            dY,
            None,
            None,
        )

    pass


pass


# [TODO] Unsure why RoPE Embedding is not torch.compiling properly
@torch.compiler.disable
def fast_rope_embedding(
    Q: torch.Tensor, K: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Fast RoPE embedding to query and key tensors.

    This function applies the optimized RoPE implementation to both query and key tensors.

    Args:
            Q (`torch.Tensor`): Query tensor
            K (`torch.Tensor`): Key tensor
            cos (`torch.Tensor`): Cosine values tensor
            sin (`torch.Tensor`): Sine values tensor

    Returns:
            tuple[torch.Tensor, torch.Tensor]: (Q, K) with RoPE embedding applied
    """
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K


pass


class Slow_RoPE_Embedding(torch.autograd.Function):
    """
    Standard RoPE embedding implementation for comparison.

    This class implements the RoPE (Rotary Positional Embedding) operation using standard PyTorch operations.
    It's provided as a reference implementation that is slower but more straightforward than the optimized version.

    RoPE is applied as: Q * cos + rotate_half(Q) * sin

    See the original paper for more details about Rotary Positional Embeddings.
    """

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of the standard RoPE embedding.

        Args:
                ctx: Context object for saving values for backward pass
                Q (`torch.Tensor`): Query tensor
                cos (`torch.Tensor`): Cosine values tensor
                sin (`torch.Tensor`): Sine values tensor
                position_ids (`Optional[torch.Tensor]`): Optional position indices

        Returns:
                `torch.Tensor`: Tensor with RoPE embedding applied
        """
        if position_ids is not None:
            # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        # Q * cos + rotate_half(Q) * sin
        half = Q.shape[-1] // 2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
        Q *= cos
        Q.addcmul_(RH_Q, sin)
        # RH_Q *= sin
        # Q += RH_Q
        ctx.save_for_backward(cos, sin)
        return Q

    pass

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """
        Backward pass of the standard RoPE embedding.

        Args:
                ctx: Context object containing saved values from forward pass
                dY (`torch.Tensor`): Gradient tensor

        Returns:
                tuple[torch.Tensor, None, None, None]: Gradient with respect to input Q, with None values for cos, sin, and position_ids
        """
        cos, sin = ctx.saved_tensors
        # Q * cos + rotate_half.T(Q) * sin
        half = dY.shape[-1] // 2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim=-1)
        dY *= cos
        dY.addcmul_(RH_dY, sin)
        # RH_dY *= sin
        # dY += RH_dY
        return dY, None, None, None

    pass


pass


def inplace_rope_embedding(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies standard RoPE embedding to query and key tensors in-place.

    This function applies the standard RoPE implementation to both query and key tensors.

    Args:
            Q (`torch.Tensor`): Query tensor
            K (`torch.Tensor`): Key tensor
            cos (`torch.Tensor`): Cosine values tensor
            sin (`torch.Tensor`): Sine values tensor
            position_ids (`Optional[torch.Tensor]`): Optional position indices

    Returns:
            tuple[torch.Tensor, torch.Tensor]: (Q, K) with RoPE embedding applied
    """
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids)
    return Q, K


pass
