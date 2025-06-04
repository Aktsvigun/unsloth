"""
Manual tuning utils
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from itertools import product
from typing import Optional, T, OrderedDict

import pandas as pd
import torch
import triton
from triton.runtime.errors import OutOfResources

from grouped_gemm.kernels.autotuning import (
	BOOLS,
	DEFAULT_K_BLOCK_SIZES,
	DEFAULT_M_BLOCK_SIZES,
	DEFAULT_N_BLOCK_SIZES,
	DEFAULT_NUM_STAGES,
	DEFAULT_NUM_WARPS,
)


@dataclass
class DeviceProperties:
	"""
	A dataclass that holds device properties for a GPU.
	
	Args:
		NUM_SM (`int`):
			Number of streaming multiprocessors.
		NUM_REGS (`int`):
			Number of registers.
		SIZE_SMEM (`int`):
			Size of shared memory.
		WARP_SIZE (`int`):
			Size of a warp.
	"""
	NUM_SM: int
	NUM_REGS: int
	SIZE_SMEM: int
	WARP_SIZE: int


_DEVICE_PROPERTIES: Optional[DeviceProperties] = None


def get_device_properties() -> DeviceProperties:
	"""
	Get the device properties for the current GPU.
	
	Returns:
		`DeviceProperties`: A dataclass containing the device properties.
	"""
	global _DEVICE_PROPERTIES
	if _DEVICE_PROPERTIES is None:
		properties = triton.runtime.driver.active.utils.get_device_properties(
			torch.cuda.current_device()
		)
		NUM_SM = properties["multiprocessor_count"]
		NUM_REGS = properties["max_num_regs"]
		SIZE_SMEM = properties["max_shared_mem"]
		WARP_SIZE = properties["warpSize"]
		_DEVICE_PROPERTIES = DeviceProperties(NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE)
	return _DEVICE_PROPERTIES


@dataclass
class KernelConfig:
	"""
	A dataclass that holds kernel configuration parameters.
	
	Args:
		BLOCK_SIZE_M (`int`):
			Block size in the M dimension.
		BLOCK_SIZE_N (`int`):
			Block size in the N dimension.
		BLOCK_SIZE_K (`int`):
			Block size in the K dimension.
		num_warps (`int`):
			Number of warps.
		num_stages (`int`):
			Number of stages.
		flatten (`bool`):
			Whether to flatten the output.
		permute_x (`bool`):
			Whether to permute the x dimension.
		permute_y (`bool`):
			Whether to permute the y dimension.
		fuse_mul_post (`bool`):
			Whether to fuse the post multiplication.
		use_tma_store (`bool`):
			Whether to use TMA store.
	"""
	BLOCK_SIZE_M: int = 32
	BLOCK_SIZE_N: int = 32
	BLOCK_SIZE_K: int = 32
	num_warps: int = 4
	num_stages: int = 2
	flatten: bool = True
	permute_x: bool = False
	permute_y: bool = False
	fuse_mul_post: bool = False
	use_tma_store: bool = False

	def to_string(self, include_tuning_params: bool = False, include_tma: bool = False) -> str:
		"""
		Convert the kernel configuration to a string.
		
		Args:
			include_tuning_params (`bool`):
				Whether to include tuning parameters.
			include_tma (`bool`):
				Whether to include TMA parameters.
		
		Returns:
			`str`: A string representation of the kernel configuration.
		"""
		s = []
		if self.permute_x:
			s.append("permute_x")
		if self.permute_y:
			s.append("permute_y")
		if include_tuning_params:
			s.append(
				f"BLOCK_SIZE_M={self.BLOCK_SIZE_M},BLOCK_SIZE_N={self.BLOCK_SIZE_N},BLOCK_SIZE_K={self.BLOCK_SIZE_K},num_warps={self.num_warps},num_stages={self.num_stages},flatten={self.flatten}"
			)
		if include_tma:
			for f in fields(self):
				if f.name.startswith("use_tma_"):
					if getattr(self, f.name):
						s.append(f.name)
		return ",".join(s)


@dataclass
class KernelConfigForward(KernelConfig):
	"""
	A dataclass that holds kernel configuration parameters for forward pass.
	
	Args:
		use_tma_load_w (`bool`):
			Whether to use TMA load for weight.
		use_tma_load_x (`bool`):
			Whether to use TMA load for input.
		use_tma_store (`bool`):
			Whether to use TMA store.
	"""
	use_tma_load_w: bool = False
	use_tma_load_x: bool = False


@dataclass
class KernelConfigBackward_dW(KernelConfig):
	"""
	A dataclass that holds kernel configuration parameters for backward pass with respect to weight.
	
	Args:
		use_tma_load_dy (`bool`):		Whether to use TMA load for gradient of output.
		use_tma_load_x (`bool`):		Whether to use TMA load for input.
	"""
	use_tma_load_dy: bool = False
	use_tma_load_x: bool = False


@dataclass
class KernelConfigBackward_dX(KernelConfig):
	"""
	A dataclass that holds kernel configuration parameters for backward pass with respect to input.
	
	Args:
		use_tma_load_dy (`bool`):		Whether to use TMA load for gradient of output.
		use_tma_load_w (`bool`):		Whether to use TMA load for weight.
	"""
	use_tma_load_dy: bool = False
	use_tma_load_w: bool = False


@dataclass
class KernelResult:
	"""
	A dataclass that holds the results of a kernel run.
	
	Args:
		torch_time (`float`):
			Time taken by PyTorch.
		triton_time (`float`):
			Time taken by Triton.
		speedup (`float`):
			Speedup of Triton over PyTorch.
		kernel_config (`KernelConfig`):
			Kernel configuration used.
	"""
	torch_time: float
	triton_time: float
	speedup: float
	kernel_config: KernelConfig

	def to_dict(self) -> OrderedDict:
		"""
		Convert the kernel result to a dictionary.
		
		Returns:
			`OrderedDict`: A dictionary containing the kernel result.
		"""
		return OrderedDict(
			**asdict(self.kernel_config),
			torch_time=self.torch_time,
			triton_time=self.triton_time,
			speedup=self.speedup,
		)

	@staticmethod
	def to_dataframe(
		results: list["KernelResult"], sort_by: str = "speedup", ascending: bool = False
	) -> pd.DataFrame:
		"""
		Convert a list of kernel results to a pandas DataFrame.
		
		Args:
			results (`list[KernelResult]`):
				List of kernel results.
			sort_by (`str`):
				Column to sort by.
			ascending (`bool`):
				Whether to sort in ascending order.
		
		Returns:
			`pd.DataFrame`: A DataFrame containing the kernel results.
		"""
		df = pd.DataFrame([result.to_dict() for result in results])
		df = df.sort_values(by=sort_by, ascending=ascending)
		return df

	@staticmethod
	def to_csv(
		results: list["KernelResult"],
		sort_by: str = "speedup",
		ascending: bool = False,
		filename: str = "results.csv",
	) -> None:
		"""
		Save a list of kernel results to a CSV file.
		
		Args:
			results (`list[KernelResult]`):
				List of kernel results.
			sort_by (`str`):
				Column to sort by.
			ascending (`bool`):
				Whether to sort in ascending order.
			filename (`str`):
				Name of the CSV file.
		"""
		df = KernelResult.to_dataframe(results, sort_by, ascending)
		df.to_csv(filename, index=False)

	@staticmethod
	def print_table(
		results: list["KernelResult"],
		sort_by: str = "speedup",
		ascending: bool = False,
		num_results: int = 10,
	) -> None:
		"""
		Print a table of kernel results.
		
		Args:
			results (`list[KernelResult]`):
				List of kernel results.
			sort_by (`str`):
				Column to sort by.
			ascending (`bool`):
				Whether to sort in ascending order.
			num_results (`int`):
				Number of results to print.
		"""
		df = KernelResult.to_dataframe(results, sort_by, ascending)
		print(df.head(num_results).to_string(index=False))


def get_kernel_configs(
	BLOCK_M: list[int]=DEFAULT_M_BLOCK_SIZES,
	BLOCK_N: list[int]=DEFAULT_N_BLOCK_SIZES,
	BLOCK_K: list[int]=DEFAULT_K_BLOCK_SIZES,
	num_warps: list[int]=DEFAULT_NUM_WARPS,
	num_stages: list[int]=DEFAULT_NUM_STAGES,
	use_tma_loads: list[bool]=BOOLS,
	fuse_permute: list[bool]=BOOLS,
) -> tuple[list[KernelConfigForward], list[KernelConfigBackward_dW], list[KernelConfigBackward_dX]]:
	"""
	Generate a list of kernel configurations.
	
	Args:
		BLOCK_M (`list[int]`):
			List of block sizes in the M dimension.
		BLOCK_N (`list[int]`):
			List of block sizes in the N dimension.
		BLOCK_K (`list[int]`):
			List of block sizes in the K dimension.
		num_warps (`list[int]`):
			List of number of warps.
		num_stages (`list[int]`):
			List of number of stages.
		use_tma_loads (`list[bool]`):
			List of whether to use TMA loads.
		fuse_permute (`list[bool]`):
			List of whether to fuse permute.
	
	Returns:
		`tuple[list[KernelConfigForward], list[KernelConfigBackward_dW], list[KernelConfigBackward_dX]]`: A tuple of lists of kernel configurations for forward, backward with respect to weight, and backward with respect to input.
	"""
	kernel_configs_fwd = []
	kernel_configs_backward_dW = []
	kernel_configs_backward_dX = []
	for block_m, block_n, block_k, w, s, use_tma_load, permute in product(
		BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, use_tma_loads, fuse_permute
	):
		kernel_configs_fwd.append(
			KernelConfigForward(
				BLOCK_SIZE_M=block_m,
				BLOCK_SIZE_N=block_n,
				BLOCK_SIZE_K=block_k,
				num_warps=w,
				num_stages=s,
				use_tma_load_x=use_tma_load,
				use_tma_load_w=use_tma_load,
				use_tma_store=False,
				permute_x=permute,
				permute_y=permute,
			)
		)
		kernel_configs_backward_dW.append(
			KernelConfigBackward_dW(
				BLOCK_SIZE_M=block_m,
				BLOCK_SIZE_N=block_n,
				BLOCK_SIZE_K=block_k,
				num_warps=w,
				num_stages=s,
				use_tma_load_dy=use_tma_load,
				use_tma_load_x=use_tma_load,
				use_tma_store=False,
				permute_x=permute,
				permute_y=permute,
			)
		)
		kernel_configs_backward_dX.append(
			KernelConfigBackward_dX(
				BLOCK_SIZE_M=block_m,
				BLOCK_SIZE_N=block_n,
				BLOCK_SIZE_K=block_k,
				num_warps=w,
				num_stages=s,
				use_tma_load_dy=use_tma_load,
				use_tma_load_w=use_tma_load,
				use_tma_store=False,
				permute_x=permute,
				permute_y=permute,
			)
		)

	kernel_configs_fwd = prune_kernel_configs_fwd(kernel_configs_fwd)
	kernel_configs_backward_dW = prune_kernel_configs_backward_dW(
		kernel_configs_backward_dW
	)
	kernel_configs_backward_dX = prune_kernel_configs_backward_dX(
		kernel_configs_backward_dX
	)
	return kernel_configs_fwd, kernel_configs_backward_dW, kernel_configs_backward_dX


def prune_kernel_configs_fwd(configs: list[KernelConfigForward]) -> list[KernelConfigForward]:
	"""
	Prune the list of forward kernel configurations.
	
	Args:
		configs (`list[KernelConfigForward]`):
			List of forward kernel configurations.
	
	Returns:
		`list[KernelConfigForward]`: Pruned list of forward kernel configurations.
	"""
	pruned_configs = []
	for config in configs:
		if config.use_tma_load_x and config.permute_x:
			continue
		if config.permute_x and config.permute_y:
			continue
		if config.use_tma_store and config.permute_y:
			continue
		pruned_configs.append(config)
	return pruned_configs


def prune_kernel_configs_backward_dX(configs: list[KernelConfigBackward_dX]) -> list[KernelConfigBackward_dX]:
	"""
	Prune the list of backward kernel configurations with respect to input.
	
	Args:
		configs (`list[KernelConfigBackward_dX]`):
			List of backward kernel configurations with respect to input.
	
	Returns:
		`list[KernelConfigBackward_dX]`: Pruned list of backward kernel configurations with respect to input.
	"""
	pruned_configs = []
	for config in configs:
		if config.use_tma_load_dy and config.permute_y:
			continue
		if config.permute_x and config.permute_y:
			continue
		if config.use_tma_store and config.permute_x:
			continue
		pruned_configs.append(config)
	return pruned_configs


def prune_kernel_configs_backward_dW(configs: list[KernelConfigBackward_dW]) -> list[KernelConfigBackward_dW]:
	"""
	Prune the list of backward kernel configurations with respect to weight.
	
	Args:
		configs (`list[KernelConfigBackward_dW]`):
			List of backward kernel configurations with respect to weight.
	
	Returns:
		`list[KernelConfigBackward_dW]`: Pruned list of backward kernel configurations with respect to weight.
	"""
	pruned_configs = []
	for config in configs:
		if config.use_tma_load_dy and config.permute_y:
			continue
		if config.use_tma_load_x and config.permute_x:
			continue
		if config.permute_x and config.permute_y:
			continue
		pruned_configs.append(config)
	return pruned_configs


class TritonTuningContext:
	"""
	A context manager for tuning Triton kernels.
	
	Args:
		kernel_config (`KernelConfig`):
			Kernel configuration to use.
	"""
	def __init__(self, kernel_config: KernelConfig):
		self.kernel_config = kernel_config
		self.success = True

	def __enter__(self) -> TritonTuningContext:
		# Setup code can be added here if needed
		return self

	def __exit__(self, exc_type: Optional[type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> bool:
		"""
		Handle exceptions raised during Triton kernel tuning.
		
		Args:
			exc_type (`type[BaseException]`):
				Type of the exception.
			exc_value (`BaseException`):
				Exception instance.
			traceback (`TracebackType`):
				Traceback of the exception.
		"""
		if exc_type is OutOfResources:
			name = exc_value.name
			required = exc_value.required
			limit = exc_value.limit
			print(
				f"Kernel config {self.kernel_config} failed: {name}, required: {required}, limit: {limit}"
			)
			self.success = False
		elif exc_type is not None:
			print(
				f"Error running Triton grouped GEMM for kernel config: {self.kernel_config}: {exc_value}"
			)
			self.success = False
		# Return False to propagate exceptions, True to suppress them
		return True