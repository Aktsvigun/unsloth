from typing import Any
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

@triton.jit
def _rms_layernorm_forward(
    Y: torch.Tensor, Y_row_stride: int,
    X: torch.Tensor, X_row_stride: int,
    W: torch.Tensor, W_row_stride: int,
    r: torch.Tensor, r_row_stride : tl.constexpr,
    n_cols     : tl.constexpr,
    eps        : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
) -> None:
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)#.to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype) # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)
pass


def _rms_layernorm_backward(
    dY: torch.Tensor, dY_row_stride: int,
    dX: torch.Tensor, dX_row_stride: int,
    X: torch.Tensor,   X_row_stride: int,
    W: torch.Tensor,   W_row_stride: int,
    r: torch.Tensor,   r_row_stride : tl.constexpr,
    # dW, dW_row_stride,
    n_cols     : tl.constexpr,
    eps        : tl.constexpr,
    GEMMA      : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
) -> None:
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    if GEMMA: dX += row_idx * dY_row_stride
    else:     dX = dY

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA: dY_W = dY_row * (W_row + 1.0)
    else:     dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask = mask)
pass
_rms_layernorm_backward = triton.jit(_rms_layernorm_backward)
_rms_layernorm_backward = triton.heuristics(
    {
        "GEMMA": lambda args: bool(args["GEMMA"]),
    }
)(_rms_layernorm_backward)


@triton.jit
def _gemma_rms_layernorm_forward(
    Y: torch.Tensor, Y_row_stride: int,
    X: torch.Tensor, X_row_stride: int,
    W: torch.Tensor, W_row_stride: int,
    r: torch.Tensor, r_row_stride : tl.constexpr,
    n_cols     : tl.constexpr,
    eps        : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
) -> None:
    # Copies https://github.com/google-deepmind/gemma/blob/main/gemma/layers.py#L31
    # and https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L33
    # exactly. Essentially all in float32!
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask = mask)
pass


class Fast_RMS_Layernorm(torch.autograd.Function):
    """
    A PyTorch autograd function for fast RMS Layer Normalization using Triton kernels.
    
    This class provides a forward and backward pass for RMS Layer Normalization optimized with Triton kernels,
    which can be significantly faster than the standard PyTorch implementation.
    
    Methods:
        forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool = False) -> torch.Tensor:
            Computes the forward pass of the RMS Layer Normalization.
            
        backward(ctx, dY: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
            Computes the backward pass of the RMS Layer Normalization.
    """
    @staticmethod
    def forward(ctx, X : torch.Tensor, W : torch.Tensor, eps : float, gemma : bool = False) -> torch.Tensor:
        """
        Computes the forward pass of the RMS Layer Normalization.
        
        Args:
            ctx: Context object used to save information for the backward pass.    X (`torch.Tensor`):
                The input tensor to be normalized.
            W (`torch.Tensor`):
                The weight tensor used in the normalization.
            eps (`float`):
                A small value added to the variance to avoid division by zero.    gemma (`bool`, optional):
                A flag indicating whether to use the Gemma version of the RMS Layer Normalization.
        
        Returns:
            `torch.Tensor`: The normalized output tensor.
        """
        shape = X.shape
        dim : int = shape[-1]
        X = X.view(-1, dim)
        n_rows : int
        n_cols : int
        n_rows, n_cols = X.shape
        BLOCK_SIZE : int
        num_warps  : int
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r = torch.empty(n_rows, dtype = torch.float32, device = device)

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        with torch_cuda_device(device):
            fx[(n_rows,)](
                Y, Y.stride(0),
                X, X.stride(0),
                W, W.stride(0),
                r, r.stride(0),
                n_cols, eps,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps  = num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)
    pass

    @staticmethod
    def backward(ctx, dY : torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """
        Computes the backward pass of the RMS Layer Normalization.
        
        Args:
            ctx: Context object containing saved information from the forward pass.    dY (`torch.Tensor`):
                The gradient of the output tensor.
        
        Returns:
            `tuple[torch.Tensor, None, None, None]`: The gradient of the input tensor and None for the other parameters.
        """
        shape = dY.shape
        dim : int = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows : int
        n_cols : int
        n_rows, n_cols = dY.shape
        # dW = X
        dX = torch.empty_like(dY) if ctx.GEMMA else dY

        with torch_cuda_device(dY.device):
            _rms_layernorm_backward[(n_rows,)](
                dY, dY.stride(0),
                dX, dX.stride(0),
                X,  X .stride(0),
                W,  W .stride(0),
                r,  r .stride(0),
                # dW, dW.stride(0),
                n_cols, ctx.eps,
                GEMMA      = ctx.GEMMA,
                BLOCK_SIZE = ctx.BLOCK_SIZE,
                num_warps  = ctx.num_warps,
            )
        dX = dX.view(*shape)
        return dX, None, None, None
    pass
pass


# [TODO] Unsure why RMS Layernorm is not torch.compiling properly
@torch.compiler.disable
def fast_rms_layernorm(layernorm: torch.nn.Module, X : torch.Tensor, gemma : bool = False) -> torch.Tensor:
    """
    A wrapper function for the Fast_RMS_Layernorm autograd function.
    
    Args:
        layernorm (`torch.nn.Module`):
            The RMS Layer Normalization module.    X (`torch.Tensor`):
            The input tensor to be normalized.    gemma (`bool`, optional):
            A flag indicating whether to use the Gemma version of the RMS Layer Normalization.
    
    Returns:
        `torch.Tensor`: The normalized output tensor.
    """
    W : torch.Tensor = layernorm.weight
    eps : float = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out
pass


from transformers.models.llama.modeling_llama import LlamaRMSNorm
class Unsloth_LlamaRMSNorm(LlamaRMSNorm):
    """
    A custom implementation of the LlamaRMSNorm class using the fast RMS Layer Normalization.
    
    This class overrides the forward method of the LlamaRMSNorm class to use the fast RMS Layer Normalization
    implemented in the Fast_RMS_Layernorm class.
    
    Methods:
        forward(self, X: torch.Tensor) -> torch.Tensor:
            Computes the forward pass of the RMS Layer Normalization using the fast implementation.
    """
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the RMS Layer Normalization using the fast implementation.
        
        Args:
            self: The instance of the Unsloth_LlamaRMSNorm class.    X (`torch.Tensor`):
                The input tensor to be normalized.
        
        Returns:
            `torch.Tensor`: The normalized output tensor.
        """
        return fast_rms_layernorm(self, X, gemma = False)
    pass
pass

try:
    from transformers.models.mllama.modeling_mllama import MllamaTextRMSNorm
    class Unsloth_MllamaTextRMSNorm(MllamaTextRMSNorm):
        def forward(self, X):
            return fast_rms_layernorm(self, X, gemma = False)
        pass
    pass
except:
    pass
pass

def patch_rms_layernorm() -> None:
    """
    Patches the LlamaRMSNorm class in the transformers library to use the fast RMS Layer Normalization.
    
    This function replaces the LlamaRMSNorm class with the Unsloth_LlamaRMSNorm class in the transformers library,
    which uses the fast RMS Layer Normalization implemented in the Fast_RMS_Layernorm class.
    
    Returns:
        None
    """
    import transformers.models.llama.modeling_llama
    transformers.models.llama.modeling_llama.LlamaRMSNorm = Unsloth_LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama
        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = Unsloth_MllamaTextRMSNorm
    except:
        pass
    return
pass


def unpatch_rms_layernorm() -> None:
    """
    Unpatches the LlamaRMSNorm class in the transformers library to revert to the original implementation.
    
    This function replaces the LlamaRMSNorm class in the transformers library with the original implementation,
    reverting the changes made by the patch_rms_layernorm function.
    
    Returns:
        None
    """
    import transformers.models.llama.modeling_llama
    transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama
        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = MllamaTextRMSNorm
    except:
        pass
    return
pass


def test_rms_layernorm(
    dim: int = 1024, eps: float = 1e-5, dtype: torch.dtype = torch.float16,
    bsz: int = 21, random_state: int = 3407, seqlen: int = 3341,
) -> None:
    """
    Tests the fast RMS Layer Normalization implementation against the standard PyTorch implementation.
    
    This function creates a test case with random input data and compares the output of the fast RMS Layer
    Normalization implementation with the output of the standard PyTorch implementation.
    
    Args:
        dim (`int`, optional):
            The dimension of the input tensor.    eps (`float`, optional):
            A small value added to the variance to avoid division by zero.    dtype (`torch.dtype`, optional):
            The data type of the input tensor.    bsz (`int`, optional):
            The batch size of the input tensor.    random_state (`int`, optional):
            The random seed used to generate the input data.    seqlen (`int`, optional):
            The sequence length of the input tensor.
    
    Returns:
        None
    """
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    XX = X.clone()
    X .requires_grad_(True)
    XX.requires_grad_(True)
    Y = layernorm(X)
    YY = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda", requires_grad = True)
    Y.backward(YY)
    correct_grad = X.grad.clone()
    # from unsloth.kernels import fast_rms_layernorm
    Y = fast_rms_layernorm(layernorm, XX)
    Y.backward(YY)
    assert(torch.amax(correct_grad - XX.grad).item() <= 0.05)
pass


def testing_suite_layernorm() -> None:
    """
    Runs a testing suite for the fast RMS Layer Normalization implementation.
    
    This function runs multiple test cases with different input dimensions, data types, sequence lengths,
    and random seeds to ensure the correctness of the fast RMS Layer Normalization implementation.
    
    Args:
        None
    
    Returns:
        None
    """
    for dim in [512, 1024, 2048]:
        for dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type = "cuda", dtype = dtype):
                for seqlen in [3341, 2048, 349]:
                    for random_state in [3407, 42]:
                        test_rms_layernorm(
                            dim = dim,
                            eps = 1e-5,
                            dtype = dtype,
                            bsz = 21,
                            random_state = random_state,
                            seqlen = seqlen,
                        )
                    pass
                pass
            pass
        pass
    pass
pass
