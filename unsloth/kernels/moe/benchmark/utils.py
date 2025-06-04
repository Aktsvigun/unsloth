from typing import Any
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
	Merges test configuration parameters with kernel results into a DataFrame.
	
	Args:
		df (`pd.DataFrame`):
			DataFrame containing kernel results.
		mode (`str`):
			Operation mode (e.g., 'forward', 'dW', 'dX').
		seqlen (`int`):
			Sequence length used in the test.
		dtype (`torch.dtype`):
			Data type used in the test.
		autotune (`bool`):
			Whether autotuning was used.
	
	Returns:
		`pd.DataFrame`: Merged DataFrame with test configuration and kernel results.
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
	Converts a list of KernelResult objects to a DataFrame and merges with test configuration.
	
	Args:
		results (`list[KernelResult]`):
			List of KernelResult objects containing benchmark results.
		mode (`str`):
			Operation mode (e.g., 'forward', 'dW', 'dX').
		seqlen (`int`):
			Sequence length used in the test.
		dtype (`torch.dtype`):
			Data type used in the test.
		autotune (`bool`):
			Whether autotuning was used.
	
	Returns:
		`pd.DataFrame`: Post-processed DataFrame containing results and test configuration.
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
	Saves benchmark results to a CSV file.
	
	Args:
		df (`pd.DataFrame`):
			DataFrame containing results to be saved.
		results_dir (`str`):
			Directory path where results should be saved.
		mode (`str`):
			Operation mode (e.g., 'forward', 'dW', 'dX').
		seqlen (`int`):
			Sequence length used in the test.
		dtype (`torch.dtype`):
			Data type used in the test.
		autotune (`bool`):
			Whether autotuning was used.
	
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


def create_kernel_configs(args: argparse.Namespace, permute_x: bool, permute_y: bool) -> list[KernelConfig]:
	"""
	Generates and prunes kernel configurations based on input arguments.
	
	Args:
		args (`argparse.Namespace`):
			Namespace containing command-line arguments for kernel configuration.
		permute_x (`bool`):
			Whether to permute input x.
		permute_y (`bool`):
			Whether to permute input y.
	
	Returns:
		`list[KernelConfig]`: List of pruned kernel configurations.
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
	Generates a list of powers of two between start and end (inclusive).
	
	Args:
		start (`int`):
			Starting value (must be a power of two).
		end (`int`):
			Ending value (must be a power of two).
	
	Returns:
		`list[int]`: List of powers of two between start and end.
	"""
	start = math.log2(start)
	end = math.log2(end)
	return [2**i for i in range(int(start), int(end) + 1)]


def multiples_of_range(start: int, end: int, step: int=1) -> list[int]:
	"""
	Generates a list of numbers from start to end with specified step.
	
	Args:
		start (`int`):
			Starting value.
		end (`int`):
			Ending value.
		step (`int`, optional, default=1):
			Step size between values.
	
	Returns:
		`list[int]`: List of numbers from start to end with specified step.
	"""
	return list(range(start, end + step, step))


def map_key_to_args(key: str, mode: str) -> None:
	"""
	Maps a key to corresponding arguments (currently unimplemented).
	
	Args:
		key (`str`):
			Key to be mapped.
		mode (`str`):
			Operation mode.
	
	Returns:
		None
	"""
	pass


def save_autotune_results(autotune_cache: dict, mode: str, ref_time: float, fused_time: float, results_dir: str) -> None:
	"""
	Saves autotuning results to a JSON file.
	
	Args:
		autotune_cache (`dict`):
			Dictionary containing autotuning results.
		mode (`str`):
			Operation mode (e.g., 'forward', 'dW', 'dX').
		ref_time (`float`):
			Reference execution time.
		fused_time (`float`):
			Fused kernel execution time.
		results_dir (`str`):
			Directory path where results should be saved.
	
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


def get_autotuner(mode: str):
	"""
	Retrieves the appropriate autotuned kernel based on mode.
	
	Args:
		mode (`str`):
			Operation mode ('forward', 'dW', 'dX', or 'backward').
	
	Returns:
		Autotuned kernel function(s) corresponding to the specified mode.
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
	Processes and saves autotuning results.
	
	Args:
		autotuner: Autotuner object containing cache of results.
		mode (`str`):
			Operation mode (e.g., 'forward', 'dW', 'dX').
		ref_time (`float`):
			Reference execution time.
		fused_time (`float`):
			Fused kernel execution time.
		results_dir (`str`):
			Directory path where results should be saved.
	
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
