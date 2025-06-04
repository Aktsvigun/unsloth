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
	Converts a value to a list if it is not already a list or None.
	
	Args:
		val (Optional[Any]): The value to convert to a list.
	
	Returns:
		Optional[List[Any]]: The value as a list, or None if the input was None.
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
		args (List[Any]): A list of values to convert to lists.
	
	Returns:
		List[List[Any]]: A list of lists, where each element is the result of applying `val_to_list` to the corresponding element in `args`.
	"""
	return [val_to_list(arg) for arg in args]


def get_forward_configs(
	BLOCK_M: List[int]=DEFAULT_M_BLOCK_SIZES,
	BLOCK_N: List[int]=DEFAULT_N_BLOCK_SIZES,
	BLOCK_K: List[int]=DEFAULT_K_BLOCK_SIZES,
	TMA_LOAD_X: List[bool]=True,
	TMA_LOAD_W: List[bool]=True,
	TMA_STORE: List[bool]=False,  # NOTE: TMA_STORE is disabled for now
	num_warps: List[int]=DEFAULT_NUM_WARPS,
	num_stages: List[int]=DEFAULT_NUM_STAGES,
	num_ctas: List[int]=DEFAULT_NUM_CTAS,
) -> List[triton.Config]:
	"""
	Generates a list of triton.Config objects for forward pass kernel configurations.
	
	Args:
		BLOCK_M (List[int], optional): List of block sizes for dimension M. Defaults to DEFAULT_M_BLOCK_SIZES.
		BLOCK_N (List[int], optional): List of block sizes for dimension N. Defaults to DEFAULT_N_BLOCK_SIZES.
		BLOCK_K (List[int], optional): List of block sizes for dimension K. Defaults to DEFAULT_K_BLOCK_SIZES.
		TMA_LOAD_X (List[bool], optional): List of boolean flags indicating whether to use TMA for loading X. Defaults to True.
		TMA_LOAD_W (List[bool], optional): List of boolean flags indicating whether to use TMA for loading W. Defaults to True.
		TMA_STORE (List[bool], optional): List of boolean flags indicating whether to use TMA for storing. Defaults to False.
		num_warps (List[int], optional): List of numbers of warps. Defaults to DEFAULT_NUM_WARPS.
		num_stages (List[int], optional): List of numbers of stages. Defaults to DEFAULT_NUM_STAGES.
		num_ctas (List[int], optional): List of numbers of CTAs. Defaults to DEFAULT_NUM_CTAS.
	
	Returns:
		List[triton.Config]: A list of triton.Config objects representing the kernel configurations.
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
	BLOCK_M: List[int]=DEFAULT_M_BLOCK_SIZES,
	BLOCK_N: List[int]=DEFAULT_N_BLOCK_SIZES,
	BLOCK_K: List[int]=DEFAULT_K_BLOCK_SIZES,
	TMA_LOAD_dY: List[bool]=True,
	TMA_LOAD_W: List[bool]=True,
	TMA_STORE: List[bool]=False,  # NOTE: TMA_STORE is disabled for now
	num_warps: List[int]=DEFAULT_NUM_WARPS,
	num_stages: List[int]=DEFAULT_NUM_STAGES,
	num_ctas: List[int]=DEFAULT_NUM_CTAS,
) -> List[triton.Config]:
	"""
	Generates a list of triton.Config objects for dX (gradient of X) kernel configurations.
	
	Args:
		BLOCK_M (List[int], optional): List of block sizes for dimension M. Defaults to DEFAULT_M_BLOCK_SIZES.
		BLOCK_N (List[int], optional): List of block sizes for dimension N. Defaults to DEFAULT_N_BLOCK_SIZES.
		BLOCK_K (List[int], optional): List of block sizes for dimension K. Defaults to DEFAULT_K_BLOCK_SIZES.
		TMA_LOAD_dY (List[bool], optional): List of boolean flags indicating whether to use TMA for loading dY. Defaults to True.
		TMA_LOAD_W (List[bool], optional): List of boolean flags indicating whether to use TMA for loading W. Defaults to True.
		TMA_STORE (List[bool], optional): List of boolean flags indicating whether to use TMA for storing. Defaults to False.
		num_warps (List[int], optional): List of numbers of warps. Defaults to DEFAULT_NUM_WARPS.
		num_stages (List[int], optional): List of numbers of stages. Defaults to DEFAULT_NUM_STAGES.
		num_ctas (List[int], optional): List of numbers of CTAs. Defaults to DEFAULT_NUM_CTAS.
	
	Returns:
		List[triton.Config]: A list of triton.Config objects representing the kernel configurations.
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
	BLOCK_M: List[int]=DEFAULT_M_BLOCK_SIZES,
	BLOCK_N: List[int]=DEFAULT_N_BLOCK_SIZES,
	BLOCK_K: List[int]=DEFAULT_K_BLOCK_SIZES,
	num_warps: List[int]=DEFAULT_NUM_WARPS,
	num_stages: List[int]=DEFAULT_NUM_STAGES,
	num_ctas: List[int]=DEFAULT_NUM_CTAS,
	TMA_LOAD_dY: List[bool]=True,
	TMA_LOAD_X: List[bool]=True,
	TMA_STORE: List[bool]=False,
) -> List[triton.Config]:
	"""
	Generates a list of triton.Config objects for dW (gradient of W) kernel configurations.
	
	Args:
		BLOCK_M (List[int], optional): List of block sizes for dimension M. Defaults to DEFAULT_M_BLOCK_SIZES.
		BLOCK_N (List[int], optional): List of block sizes for dimension N. Defaults to DEFAULT_N_BLOCK_SIZES.
		BLOCK_K (List[int], optional): List of block sizes for dimension K. Defaults to DEFAULT_K_BLOCK_SIZES.
		num_warps (List[int], optional): List of numbers of warps. Defaults to DEFAULT_NUM_WARPS.
		num_stages (List[int], optional): List of numbers of stages. Defaults to DEFAULT_NUM_STAGES.
		num_ctas (List[int], optional): List of numbers of CTAs. Defaults to DEFAULT_NUM_CTAS.
		TMA_LOAD_dY (List[bool], optional): List of boolean flags indicating whether to use TMA for loading dY. Defaults to True.
		TMA_LOAD_X (List[bool], optional): List of boolean flags indicating whether to use TMA for loading X. Defaults to True.
		TMA_STORE (List[bool], optional): List of boolean flags indicating whether to use TMA for storing. Defaults to False.
	
	Returns:
		List[triton.Config]: A list of triton.Config objects representing the kernel configurations.
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
	Estimates the shared memory requirements for a given kernel configuration.
	
	Args:
		num_stages (int): Number of stages in the kernel.
		BLOCK_SIZE_M (int): Block size for dimension M.
		BLOCK_SIZE_N (int): Block size for dimension N.
		BLOCK_SIZE_K (int): Block size for dimension K.
		dtype (torch.dtype): Data type of the tensors.
	
	Returns:
		int: Estimated shared memory requirements in bytes.
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
	Checks if the shared memory requirements exceed the device's capacity.
	
	Args:
		num_stages (int): Number of stages in the kernel.
		BLOCK_SIZE_M (int): Block size for dimension M.
		BLOCK_SIZE_N (int): Block size for dimension N.
		BLOCK_SIZE_K (int): Block size for dimension K.
		dtype (torch.dtype): Data type of the tensors.
		smem_size (int): Available shared memory size on the device.
		slack (float, optional): Slack space to account for other memory requirements. Defaults to 50000.
	
	Returns:
		bool: True if the shared memory requirements exceed the device's capacity, False otherwise.
	"""
	smem_reqs = estimate_smem_reqs(
		num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype
	)
	return smem_reqs > smem_size + slack


def common_prune_criteria(config: triton.Config, kwargs: dict, dtype: torch.dtype) -> bool:
	"""
	Common criteria for pruning kernel configurations based on various constraints.
	
	Args:
		config (triton.Config): The kernel configuration to evaluate.
		kwargs (dict): Additional arguments containing context information.
		dtype (torch.dtype): Data type of the tensors.
	
	Returns:
		bool: True if the configuration should be pruned, False otherwise.
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
	#	 return True
	return False


def maybe_disable_tma(config: triton.Config) -> None:
	"""
	Disables TMA (Tensor Memory Acceleration) in the configuration if the device does not support it.
	
	Args:
		config (triton.Config): The kernel configuration to modify.
	
	Returns:
		None: The configuration is modified in place.
	"""
	from grouped_gemm.interface import supports_tma

	tma_keys = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
	if not supports_tma():
		logger.info("Disabling TMA")
		for k in tma_keys:
			config.kwargs[k] = False


def prune_kernel_configs_fwd(configs: list[triton.Config], args: List[Any], **kwargs) -> List[triton.Config]:
	"""
	Prunes forward pass kernel configurations based on specific criteria.
	
	Args:
		configs (list[triton.Config]): List of kernel configurations to prune.
		args (List[Any]): Additional arguments.
		kwargs (dict): Additional keyword arguments containing context information.
	
	Returns:
		List[triton.Config]: A list of pruned kernel configurations.
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


def prune_dX_configs(configs: List[triton.Config], args: List[Any], **kwargs) -> List[triton.Config]:
	"""
	Prunes dX (gradient of X) kernel configurations based on specific criteria.
	
	Args:
		configs (List[triton.Config]): List of kernel configurations to prune.
		args (List[Any]): Additional arguments.
		kwargs (dict): Additional keyword arguments containing context information.
	
	Returns:
		List[triton.Config]: A list of pruned kernel configurations.
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


def prune_kernel_configs_backward_dW(configs: list[triton.Config], args: List[Any], **kwargs) -> List[triton.Config]:
	"""
	Prunes dW (gradient of W) kernel configurations based on specific criteria.
	
	Args:
		configs (list[triton.Config]): List of kernel configurations to prune.
		args (List[Any]): Additional arguments.
		kwargs (dict): Additional keyword arguments containing context information.
	
	Returns:
		List[triton.Config]: A list of pruned kernel configurations.
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