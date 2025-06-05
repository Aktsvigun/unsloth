from typing import Any, Union
import argparse
import datetime
import json
import logging
import math
import os
from itertools import product

import pandas as pd
import torch

from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
    KernelResult,
)

SEED = 42


def create_merged_results(
    df: pd.DataFrame, mode: str, seqlen: int, dtype: torch.dtype, autotune: bool
) -> pd.DataFrame:
    """
    Merges a dataframe with kernel results with test configuration data.
    
    Args:
            df (`pd.DataFrame`):
                DataFrame containing kernel results.
            mode (`str`):
                The mode of the test (e.g., 'forward', 'dW', 'dX').
            seqlen (`int`):
                The sequence length used in the test.
            dtype (`torch.dtype`):
                The data type used in the test.
            autotune (`bool`):
                Whether autotuning was used in the test.
    
        Returns:
            `pd.DataFrame`:
                The merged dataframe with test configuration columns added.
    """
    kernel_result_cols = df.columns.to_list()
    test_config_dict = {
        "mode": mode,
        "seqlen": seqlen,
        "dtype": dtype,
        "autotune": autotune,
    }
    test_config_cols = list(test_config_dict.keys())
    for col in test_config_cols:
        df[col] = test_config_dict[col]
    # Reorder columns so that test config cols are first
    df = df[test_config_cols + kernel_result_cols]
    return df


def post_process_results(
    results: list[KernelResult],
    mode: str,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
) -> pd.DataFrame:
    """
    Post-processes a list of kernel results into a dataframe.
    
    Args:
            results (`list[KernelResult]`):
                List of kernel results to process.
            mode (`str`):
                The mode of the test (e.g., 'forward', 'dW', 'dX').
            seqlen (`int`):
                The sequence length used in the test.
            dtype (`torch.dtype`):
                The data type used in the test.
            autotune (`bool`):
                Whether autotuning was used in the test.
    
        Returns:
            `pd.DataFrame`:
                The post-processed dataframe containing the results.
    """
    df = KernelResult.to_dataframe(results, sort_by="speedup")
    df = create_merged_results(df, mode, seqlen, dtype, autotune)
    return df


def save_results(
    df: pd.DataFrame,
    results_dir: str,
    mode: str,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
) -> None:
    """
    Saves the results dataframe to a CSV file.
    
    Args:
            df (`pd.DataFrame`):
                DataFrame containing the results to save.
            results_dir (`str`):
                Directory to save the results.
            mode (`str`):
                The mode of the test (e.g., 'forward', 'dW', 'dX').
            seqlen (`int`):
                The sequence length used in the test.
            dtype (`torch.dtype`):
                The data type used in the test.
            autotune (`bool`):
                Whether autotuning was used in the test.
    
        Returns:
            None
    """
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"{results_dir}/{mode}"
    save_path = f"{save_dir}/{dt}_{seqlen}_{str(dtype).split('.')[-1]}.csv"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving results to {save_path}")
    df.to_csv(save_path, index=False)


def create_kernel_configs(args: argparse.Namespace, permute_x: bool, permute_y: bool) -> list[Union[KernelConfigForward, KernelConfigBackward_dW, KernelConfigBackward_dX]]:
    """
    Generates a list of kernel configurations based on the given arguments.
    
    Args:
            args (`argparse.Namespace`):
                Namespace object containing the arguments for the kernel configurations.
            permute_x (`bool`):
                Whether to permute the x dimension.
            permute_y (`bool`):
                Whether to permute the y dimension.
    
        Returns:
            list[Union[KernelConfigForward, KernelConfigBackward_dW, KernelConfigBackward_dX]]:
                List of kernel configurations.
    """
    block_m_range = power_of_two_range(args.BLOCK_SIZE_M[0], args.BLOCK_SIZE_M[1])
    block_n_range = power_of_two_range(args.BLOCK_SIZE_N[0], args.BLOCK_SIZE_N[1])
    block_k_range = power_of_two_range(args.BLOCK_SIZE_K[0], args.BLOCK_SIZE_K[1])
    num_warps_range = multiples_of_range(args.num_warps[0], args.num_warps[1], step=2)
    num_stages_range = multiples_of_range(
        args.num_stages[0], args.num_stages[1], step=1
    )

    mode = args.mode
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        tma_load_a,
        tma_load_b,
    ) in product(
        block_m_range,
        block_n_range,
        block_k_range,
        num_warps_range,
        num_stages_range,
        [True, False],
        [True, False],
    ):
        if mode == "forward":
            kernel_config = KernelConfigForward(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_w=tma_load_a,
                use_tma_load_x=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        elif mode == "dW":
            kernel_config = KernelConfigBackward_dW(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_dy=tma_load_a,
                use_tma_load_x=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        elif mode == "dX":
            kernel_config = KernelConfigBackward_dX(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_dy=tma_load_a,
                use_tma_load_w=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        kernel_configs.append(kernel_config)

    logging.info(f"Pruning {len(kernel_configs)} kernel configs")

    pruned_configs = []
    for config in kernel_configs:
        if mode == "forward":
            if permute_x and config.use_tma_load_x:
                continue
        elif mode == "dW":
            if permute_x and config.use_tma_load_x:
                continue
            if permute_y and config.use_tma_load_dy:
                continue
        elif mode == "dX":
            if permute_y and config.use_tma_load_dy:
                continue
        pruned_configs.append(config)
    logging.info(f"After pruning, {len(pruned_configs)} kernel configs")

    return pruned_configs


def power_of_two_range(start: int, end: int) -> list[int]:
    """
    Generates a list of integers that are powers of two within the given range.
    
    Args:
            start (`int`):
                The start of the range (inclusive).
            end (`int`):
                The end of the range (inclusive).
    
        Returns:
            list[int]:
                List of integers that are powers of two within the given range.
    """
    start = math.log2(start)
    end = math.log2(end)
    return [2**i for i in range(int(start), int(end) + 1)]


def multiples_of_range(start: int, end: int, step: int=1) -> list[int]:
    """
    Generates a list of integers that are multiples of a given step within the given range.
    
    Args:
            start (`int`):
                The start of the range (inclusive).
            end (`int`):
                The end of the range (inclusive).
            step (`int`, optional):
                The step size.
    
        Returns:
            list[int]:
                List of integers that are multiples of the step within the given range.
    """
    return list(range(start, end + step, step))


def map_key_to_args(key: tuple[Any], mode: str) -> dict:
    pass


def save_autotune_results(autotune_cache: dict[tuple[Any], Any], mode: str, ref_time: float, fused_time: float, results_dir: str) -> None:
    """
    Saves the autotune results to JSON files.
    
    Args:
            autotune_cache (`dict[tuple[Any], Any]`):
                Dictionary of autotune results.
            mode (`str`):
                The mode of the test (e.g., 'forward', 'dW', 'dX').
            ref_time (`float`):
                The reference time.
            fused_time (`float`):
                The fused time.
            results_dir (`str`):
                Directory to save the results.
    
        Returns:
            None
    """
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"{results_dir}/{mode}/autotune/{dt}/{device_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, config in autotune_cache.items():
        key = [
            str(k) if not "torch" in str(k) else str(k.split("torch.")[-1]) for k in key
        ]
        filename = "_".join(key)
        save_path = f"{save_dir}/{filename}.json"
        print(f"Saving autotune results to {save_path}")
        with open(save_path, "w") as f:
            result = {
                **config.all_kwargs(),
                "ref_time": ref_time,
                "fused_time": fused_time,
            }
            json.dump(result, f)


def get_autotuner(mode: str) -> Union[Any, tuple[Any, Any]]:
    """
    Returns the appropriate autotuner based on the given mode.
    
    Args:
            mode (`str`):
                The mode of the test (e.g., 'forward', 'dW', 'dX', 'backward').
    
        Returns:
            Union[Any, tuple[Any, Any]]:
                The autotuner or a tuple of autotuners for the given mode.
    """
    if mode == "forward":
        from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel

        return _autotuned_grouped_gemm_forward_kernel
    elif mode == "dW":
        from grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dW_kernel

        return _autotuned_grouped_gemm_dW_kernel
    elif mode == "dX":
        from grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dX_kernel

        return _autotuned_grouped_gemm_dX_kernel
    elif mode == "backward":
        from grouped_gemm.kernels.backward import (
            _autotuned_grouped_gemm_dW_kernel,
            _autotuned_grouped_gemm_dX_kernel,
        )

        return _autotuned_grouped_gemm_dW_kernel, _autotuned_grouped_gemm_dX_kernel
    else:
        raise ValueError(f"Invalid mode: {mode}")


def postprocess_autotune_results(autotuner, mode: str, ref_time: float, fused_time: float, results_dir: str) -> None:
    """
    Post-processes and saves the autotune results.
    
    Args:
            autotuner: The autotuner object containing the results.
            mode (`str`):
                The mode of the test (e.g., 'forward', 'dW', 'dX').
            ref_time (`float`):
                The reference time.
            fused_time (`float`):
                The fused time.
            results_dir (`str`):
                Directory to save the results.
    
        Returns:
            None
    """
    for key, value in autotuner.cache.items():
        print(f"{mode} {key}: {value.all_kwargs()}")
    save_autotune_results(
        autotuner.cache,
        mode=mode,
        ref_time=ref_time,
        fused_time=fused_time,
        results_dir=results_dir,
    )