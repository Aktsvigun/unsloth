"""
Autotuning utils
"""

import logging
from itertools import product
from typing import List, Optional, Any

import torch
import triton

logger = logging.getLogger(__name__)

DEFAULT_M_BLOCK_SIZES = [64, 128]
DEFAULT_N_BLOCK_SIZES = [64, 128, 256]
DEFAULT_K_BLOCK_SIZES = [64, 128, 256]
DEFAULT_NUM_CTAS = 1
DEFAULT_NUM_WARPS = [4, 8]
DEFAULT_NUM_STAGES = [3, 4, 5]
BOOLS = [True, False]


def val_to_list(val: Optional[Any]) -> Optional[List[Any]]:
    """
    Converts a value to a list if it is not already a list.

    Args:
            val (`Optional[Any]`): The value to convert.

    Returns:
            `Optional[List[Any]]`: A list containing the value, or the original list if val is already a list. Returns None if val is None.
    """
    if val is None:
        return None
    elif isinstance(val, list):
        return val
    else:
        return [val]


def convert_args_to_list(args: List[Any]) -> List[List[Any]]:
    """
    Converts a list of values to lists using `val_to_list`.

    Args:
            args (`List[Any]`): List of values to convert.

    Returns:
            `List[List[Any]]`: List of lists where each value has been converted using `val_to_list`.
    """
    return [val_to_list(arg) for arg in args]


def get_forward_configs(
    BLOCK_M: List[int] = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: List[int] = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: List[int] = DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_X: List[bool] = True,
    TMA_LOAD_W: List[bool] = True,
    TMA_STORE: List[bool] = False,  # NOTE: TMA_STORE is disabled for now
    num_warps: List[int] = DEFAULT_NUM_WARPS,
    num_stages: List[int] = DEFAULT_NUM_STAGES,
    num_ctas: List[int] = DEFAULT_NUM_CTAS,
) -> List[triton.Config]:
    """
    Generates triton kernel configurations for forward pass.

    Args:
            BLOCK_M (`List[int]`): List of block sizes for M dimension.
            BLOCK_N (`List[int]`): List of block sizes for N dimension.
            BLOCK_K (`List[int]`): List of block sizes for K dimension.
            TMA_LOAD_X (`List[bool]`): List of boolean flags for TMA load X.
            TMA_LOAD_W (`List[bool]`): List of boolean flags for TMA load W.
            TMA_STORE (`List[bool]`): List of boolean flags for TMA store.
            num_warps (`List[int]`): List of warp counts.
            num_stages (`List[int]`): List of pipeline stage counts.
            num_ctas (`List[int]`): List of CTA counts.

    Returns:
            `List[triton.Config]`: List of generated triton configurations.
    """
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TMA_LOAD_X,
        TMA_LOAD_W,
        TMA_STORE,
        num_warps,
        num_stages,
        num_ctas,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            TMA_LOAD_X,
            TMA_LOAD_W,
            TMA_STORE,
            num_warps,
            num_stages,
            num_ctas,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_x,
        tma_load_w,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_X,
        TMA_LOAD_W,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_X=tma_load_x,
                    USE_TMA_LOAD_W=tma_load_w,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def get_dX_kernel_configs(
    BLOCK_M: List[int] = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: List[int] = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: List[int] = DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_dY: List[bool] = True,
    TMA_LOAD_W: List[bool] = True,
    TMA_STORE: List[bool] = False,  # NOTE: TMA_STORE is disabled for now
    num_warps: List[int] = DEFAULT_NUM_WARPS,
    num_stages: List[int] = DEFAULT_NUM_STAGES,
    num_ctas: List[int] = DEFAULT_NUM_CTAS,
) -> List[triton.Config]:
    """
    Generates triton kernel configurations for dX backward pass.

    Args:
            BLOCK_M (`List[int]`): List of block sizes for M dimension.
            BLOCK_N (`List[int]`): List of block sizes for N dimension.
            BLOCK_K (`List[int]`): List of block sizes for K dimension.
            TMA_LOAD_dY (`List[bool]`): List of boolean flags for TMA load dY.
            TMA_LOAD_W (`List[bool]`): List of boolean flags for TMA load W.
            TMA_STORE (`List[bool]`): List of boolean flags for TMA store.
            num_warps (`List[int]`): List of warp counts.
            num_stages (`List[int]`): List of pipeline stage counts.
            num_ctas (`List[int]`): List of CTA counts.

    Returns:
            `List[triton.Config]`: List of generated triton configurations.
    """
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TMA_LOAD_dY,
        TMA_LOAD_W,
        TMA_STORE,
        num_warps,
        num_stages,
        num_ctas,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            TMA_LOAD_dY,
            TMA_LOAD_W,
            TMA_STORE,
            num_warps,
            num_stages,
            num_ctas,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_dy,
        tma_load_w,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_dY,
        TMA_LOAD_W,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_dY=tma_load_dy,
                    USE_TMA_LOAD_W=tma_load_w,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def get_dW_kernel_configs(
    BLOCK_M: List[int] = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: List[int] = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: List[int] = DEFAULT_K_BLOCK_SIZES,
    num_warps: List[int] = DEFAULT_NUM_WARPS,
    num_stages: List[int] = DEFAULT_NUM_STAGES,
    num_ctas: List[int] = DEFAULT_NUM_CTAS,
    TMA_LOAD_dY: List[bool] = True,
    TMA_LOAD_X: List[bool] = True,
    TMA_STORE: List[bool] = False,
) -> List[triton.Config]:
    """
    Generates triton kernel configurations for dW backward pass.

    Args:
            BLOCK_M (`List[int]`): List of block sizes for M dimension.
            BLOCK_N (`List[int]`): List of block sizes for N dimension.
            BLOCK_K (`List[int]`): List of block sizes for K dimension.
            num_warps (`List[int]`): List of warp counts.
            num_stages (`List[int]`): List of pipeline stage counts.
            num_ctas (`List[int]`): List of CTA counts.
            TMA_LOAD_dY (`List[bool]`): List of boolean flags for TMA load dY.
            TMA_LOAD_X (`List[bool]`): List of boolean flags for TMA load X.
            TMA_STORE (`List[bool]`): List of boolean flags for TMA store.

    Returns:
            `List[triton.Config]`: List of generated triton configurations.
    """
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        num_ctas,
        TMA_LOAD_dY,
        TMA_LOAD_X,
        TMA_STORE,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps,
            num_stages,
            num_ctas,
            TMA_LOAD_dY,
            TMA_LOAD_X,
            TMA_STORE,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_dy,
        tma_load_x,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_dY,
        TMA_LOAD_X,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_dY=tma_load_dy,
                    USE_TMA_LOAD_X=tma_load_x,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def estimate_smem_reqs(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
) -> int:
    """
    Estimates shared memory requirements for a kernel configuration.

    Args:
            num_stages (`int`): Number of pipeline stages.
            BLOCK_SIZE_M (`int`): Block size for M dimension.
            BLOCK_SIZE_N (`int`): Block size for N dimension.
            BLOCK_SIZE_K (`int`): Block size for K dimension.
            dtype (`torch.dtype`): Data type of the tensors.

    Returns:
            `int`: Estimated shared memory requirements in bytes.
    """
    num_bytes = dtype.itemsize
    return (
        num_stages * BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N)
        + BLOCK_SIZE_M * BLOCK_SIZE_N
    ) * num_bytes


def exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
    smem_size: int,
    slack: float = 50000,
) -> bool:
    """
    Checks if a kernel configuration exceeds shared memory capacity.

    Args:
            num_stages (`int`): Number of pipeline stages.
            BLOCK_SIZE_M (`int`): Block size for M dimension.
            BLOCK_SIZE_N (`int`): Block size for N dimension.
            BLOCK_SIZE_K (`int`): Block size for K dimension.
            dtype (`torch.dtype`): Data type of the tensors.
            smem_size (`int`): Available shared memory size.
            slack (`float`): Slack space to account for other memory requirements.

    Returns:
            `bool`: True if configuration exceeds shared memory capacity, False otherwise.
    """
    smem_reqs = estimate_smem_reqs(
        num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype
    )
    return smem_reqs > smem_size + slack


def common_prune_criteria(
    config: triton.Config, kwargs: dict, dtype: torch.dtype
) -> bool:
    """
    Common criteria for pruning kernel configurations.

    Args:
            config (`triton.Config`): Kernel configuration to evaluate.
            kwargs (`dict`): Additional parameters for evaluation.
            dtype (`torch.dtype`): Data type of the tensors.

    Returns:
            `bool`: True if configuration should be pruned, False otherwise.
    """
    from grouped_gemm.interface import supports_tma
    from grouped_gemm.kernels.tuning import get_device_properties

    smem_size = get_device_properties().SIZE_SMEM

    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]

    num_tokens = kwargs["NUM_TOKENS"]
    num_experts = kwargs["NUM_EXPERTS"]
    permute_x = kwargs["PERMUTE_X"]
    permute_y = kwargs["PERMUTE_Y"]
    tokens_per_expert = num_tokens // num_experts

    # use_tma = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    MIN_BLOCK_SIZE_M = DEFAULT_M_BLOCK_SIZES[0]
    if exceeds_smem_capacity(
        num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, smem_size
    ):
        return True
    if BLOCK_SIZE_M > tokens_per_expert * 2 and tokens_per_expert > MIN_BLOCK_SIZE_M:
        return True
    if permute_x and permute_y:
        return True
    # if not supports_tma() and any(use_tma):
    # 	 return True
    return False


def maybe_disable_tma(config: triton.Config) -> None:
    """
    Disables TMA in configuration if not supported by hardware.

    Args:
            config (`triton.Config`): Kernel configuration to modify.

    Returns:
            None: Modifies the configuration in place.
    """
    from grouped_gemm.interface import supports_tma

    tma_keys = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    if not supports_tma():
        logger.info("Disabling TMA")
        for k in tma_keys:
            config.kwargs[k] = False


def prune_kernel_configs_fwd(
    configs: list[triton.Config], args: List[Any], **kwargs
) -> List[triton.Config]:
    """
    Prunes forward kernel configurations based on criteria.

    Args:
            configs (`list[triton.Config]`): List of kernel configurations to prune.
            args (`List[Any]`): Additional arguments.
            kwargs (`dict`): Additional parameters for pruning criteria.

    Returns:
            `List[triton.Config]`: Pruned list of kernel configurations.
    """
    x = kwargs["x_ptr"]
    dtype = x.dtype

    logger.debug(f"Pruning configs: {len(configs)}")

    pruned_configs = []
    for config in configs:
        # disable TMA if gpu does not support it
        maybe_disable_tma(config)

        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            # Dynamically disable TMA_LOAD_X for permuted X
            config.kwargs["USE_TMA_LOAD_X"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_Y"]:
            continue

        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs


def prune_dX_configs(
    configs: List[triton.Config], args: List[Any], **kwargs
) -> List[triton.Config]:
    """
    Prunes dX backward kernel configurations based on criteria.

    Args:
            configs (`List[triton.Config]`): List of kernel configurations to prune.
            args (`List[Any]`): Additional arguments.
            kwargs (`dict`): Additional parameters for pruning criteria.

    Returns:
            `List[triton.Config]`: Pruned list of kernel configurations.
    """
    dtype = kwargs["w_ptr"].dtype

    logger.debug(f"Pruning configs: {len(configs)}")
    pruned_configs = []

    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            # dynamically disable TMA_LOAD_dY for permuted Y
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_X"]:
            continue
        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs


def prune_kernel_configs_backward_dW(
    configs: list[triton.Config], args: List[Any], **kwargs
) -> List[triton.Config]:
    """
    Prunes dW backward kernel configurations based on criteria.

    Args:
            configs (`list[triton.Config]`): List of kernel configurations to prune.
            args (`List[Any]`): Additional arguments.
            kwargs (`dict`): Additional parameters for pruning criteria.

    Returns:
            `List[triton.Config]`: Pruned list of kernel configurations.
    """
    dtype = kwargs["x_ptr"].dtype

    pruned_configs = []
    logger.debug(f"Pruning configs: {len(configs)}")

    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            config.kwargs["USE_TMA_LOAD_X"] = False
        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs
