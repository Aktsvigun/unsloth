"""
Manual tuning utils
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from itertools import product
from typing import Optional, OrderedDict

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
    Represents the properties of a CUDA device.
    
    Args:
        NUM_SM (`int`):
            The number of streaming multiprocessors (SMs) on the device.
        NUM_REGS (`int`):
            The maximum number of registers per thread on the device.
        SIZE_SMEM (`int`):
            The maximum amount of shared memory per SM on the device.
        WARP_SIZE (`int`):
            The number of threads per warp on the device.
    """
    NUM_SM: int
    NUM_REGS: int
    SIZE_SMEM: int
    WARP_SIZE: int


_DEVICE_PROPERTIES: Optional[DeviceProperties] = None


def get_device_properties() -> DeviceProperties:
    """
    Retrieves and returns the properties of the current CUDA device.
    
    Returns:
        `DeviceProperties`: An instance of the `DeviceProperties` class containing the properties of the current CUDA device.
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
    Base class for kernel configurations used in Triton-based GEMM operations.
    
    Args:
        BLOCK_SIZE_M (`int`):
            The block size along the M dimension.
        BLOCK_SIZE_N (`int`):
            The block size along the N dimension.
        BLOCK_SIZE_K (`int`):
            The block size along the K dimension.
        num_warps (`int`):
            The number of warps to use per thread block.
        num_stages (`int`):
            The number of stages in the pipeline.
        flatten (`bool`):
            Whether to flatten the output tensor.
        permute_x (`bool`):
            Whether to permute the input tensor along the x dimension.
        permute_y (`bool`):
            Whether to permute the input tensor along the y dimension.
        fuse_mul_post (`bool`):
            Whether to fuse the multiplication and post-processing operations.
        use_tma_store (`bool`):
            Whether to use TMA (Tensor Memory Access) for storing the output tensor.
    """
    BLOCK_SIZE_M: int   = 32

    BLOCK_SIZE_N: int   = 32

    BLOCK_SIZE_K: int   = 32

    num_warps: int      = 4

    num_stages: int     = 2

    flatten: bool       = True

    permute_x: bool     = False

    permute_y: bool     = False

    fuse_mul_post: bool = False

    use_tma_store: bool = False


    def to_string(self, include_tuning_params: bool = False, include_tma: bool = False) -> str:
        """
        Generates a string representation of the kernel configuration.
        
        Args:
            include_tuning_params (`bool`, optional):
                Whether to include the tuning parameters (block sizes, num_warps, num_stages, etc.) in the string.
            include_tma (`bool`, optional):
                Whether to include the TMA-related parameters in the string.
        
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
    Represents a kernel configuration for the forward pass of a GEMM operation.
    
    Args:
        use_tma_load_w (`bool`):
            Whether to use TMA for loading the weight tensor.
        use_tma_load_x (`bool`):
            Whether to use TMA for loading the input tensor.
        
        All other arguments are inherited from the `KernelConfig` class.
    """
    use_tma_load_w: bool = False

    use_tma_load_x: bool = False



@dataclass
class KernelConfigBackward_dW(KernelConfig):
    """
    Represents a kernel configuration for the backward pass with respect to the weight tensor (dW) in a GEMM operation.
    
    Args:
        use_tma_load_dy (`bool`):
            Whether to use TMA for loading the gradient of the output tensor (dy).
        use_tma_load_x (`bool`):
            Whether to use TMA for loading the input tensor (x).
        
        All other arguments are inherited from the `KernelConfig` class.
    """
    use_tma_load_dy: bool = False

    use_tma_load_x: bool  = False



@dataclass
class KernelConfigBackward_dX(KernelConfig):
    """
    Represents a kernel configuration for the backward pass with respect to the input tensor (dX) in a GEMM operation.
    
    Args:
        use_tma_load_dy (`bool`):
            Whether to use TMA for loading the gradient of the output tensor (dy).
        use_tma_load_w (`bool`):
            Whether to use TMA for loading the weight tensor (w).
        
        All other arguments are inherited from the `KernelConfig` class.
    """
    use_tma_load_dy: bool = False

    use_tma_load_w: bool  = False



@dataclass
class KernelResult:
    """
    Represents the result of a kernel tuning experiment.
    
    Args:
        torch_time (`float`):
            The time taken by the PyTorch implementation in seconds.
        triton_time (`float`):
            The time taken by the Triton implementation in seconds.
        speedup (`float`):
            The speedup factor of the Triton implementation over the PyTorch implementation.    kernel_config (`KernelConfig`):
            The kernel configuration used for the experiment.
    """
    torch_time: float
    triton_time: float
    speedup: float
    kernel_config: KernelConfig

    def to_dict(self) -> OrderedDict:
        """
        Converts the kernel result to a dictionary.
        
        Returns:
            `OrderedDict`: A dictionary containing the kernel configuration parameters and the benchmark results.
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
        Converts a list of kernel results to a pandas DataFrame.
        
        Args:
            results (`list[KernelResult]`):
                A list of `KernelResult` objects.    sort_by (`str`, optional):
                The column to sort the DataFrame by.    ascending (`bool`, optional):
                Whether to sort the DataFrame in ascending order.
        
        Returns:
            `pd.DataFrame`: A pandas DataFrame containing the kernel results.
        """
        df = pd.DataFrame([result.to_dict() for result in results])
        df = df.sort_values(by=sort_by, ascending=ascending)
        return df

    @staticmethod
    def to_csv(
        results: list["KernelResult"],
        sort_by: str    = "speedup",
        ascending: bool = False,
        filename: str   = "results.csv",
    ) -> None:
        """
        Saves a list of kernel results to a CSV file.
        
        Args:
            results (`list[KernelResult]`):
                A list of `KernelResult` objects.    sort_by (`str`, optional):
                The column to sort the results by before saving.    ascending (`bool`, optional):
                Whether to sort the results in ascending order.    filename (`str`, optional):
                The name of the CSV file to save the results to.
        """
        df = KernelResult.to_dataframe(results, sort_by, ascending)
        df.to_csv(filename, index=False)

    @staticmethod
    def print_table(
        results: list["KernelResult"],
        sort_by: str     = "speedup",
        ascending: bool  = False,
        num_results: int = 10,
    ) -> None:
        """
        Prints a table of the top kernel results to the console.
        
        Args:
            results (`list[KernelResult]`):
                A list of `KernelResult` objects.    sort_by (`str`, optional):
                The column to sort the results by.    ascending (`bool`, optional):
                Whether to sort the results in ascending order.    num_results (`int`, optional):
                The number of results to print.
        """
        df = KernelResult.to_dataframe(results, sort_by, ascending)
        print(df.head(num_results).to_string(index=False))


def get_kernel_configs(
    BLOCK_M: list[int]        = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: list[int]        = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: list[int]        = DEFAULT_K_BLOCK_SIZES,
    num_warps: list[int]      = DEFAULT_NUM_WARPS,
    num_stages: list[int]     = DEFAULT_NUM_STAGES,
    use_tma_loads: list[bool] = BOOLS,
    fuse_permute: list[bool]  = BOOLS,
) -> tuple[list[KernelConfigForward], list[KernelConfigBackward_dW], list[KernelConfigBackward_dX]]:
    """
    Generates a list of kernel configurations for forward, backward dW, and backward dX passes.
    
    Args:
        BLOCK_M (`list[int]`, optional):
            A list of block sizes for the M dimension.    BLOCK_N (`list[int]`, optional):
            A list of block sizes for the N dimension.    BLOCK_K (`list[int]`, optional):
            A list of block sizes for the K dimension.    num_warps (`list[int]`, optional):
            A list of numbers of warps to use per thread block.    num_stages (`list[int]`, optional):
            A list of numbers of pipeline stages to use.    use_tma_loads (`list[bool]`, optional):
            A list of flags indicating whether to use TMA for loading.    fuse_permute (`list[bool]`, optional):
            A list of flags indicating whether to permute the input tensors.
    
    Returns:
        `tuple[list[KernelConfigForward], list[KernelConfigBackward_dW], list[KernelConfigBackward_dX]]`:
            A tuple containing lists of kernel configurations for the forward, backward dW, and backward dX passes.
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
    Prunes the list of forward kernel configurations by removing invalid combinations.
    
    Args:
        configs (`list[KernelConfigForward]`):
            A list of forward kernel configurations.
    
    Returns:
        `list[KernelConfigForward]`:
            A pruned list of forward kernel configurations.
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
    Prunes the list of backward dX kernel configurations by removing invalid combinations.
    
    Args:
        configs (`list[KernelConfigBackward_dX]`):
            A list of backward dX kernel configurations.
    
    Returns:
        `list[KernelConfigBackward_dX]`:
            A pruned list of backward dX kernel configurations.
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
    Prunes the list of backward dW kernel configurations by removing invalid combinations.
    
    Args:
        configs (`list[KernelConfigBackward_dW]`):
            A list of backward dW kernel configurations.
    
    Returns:
        `list[KernelConfigBackward_dW]`:
            A pruned list of backward dW kernel configurations.
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
    Context manager for tuning Triton kernels.
    
    Args:
        kernel_config (`KernelConfig`):
            The kernel configuration to use for tuning.    success (`bool`):
            A flag indicating whether the kernel tuning was successful.
    """
    def __init__(self, kernel_config: KernelConfig):
        self.kernel_config = kernel_config
        self.success = True

    def __enter__(self) -> TritonTuningContext:
        """
        Enters the context manager.
        
        Returns:
            `TritonTuningContext`:
                The context manager instance.
        """
        # Setup code can be added here if needed
        return self

    def __exit__(self, exc_type: type[OutOfResources] | None, exc_value: OutOfResources | None, traceback: TracebackType | None) -> bool:
        """
        Exits the context manager and handles any exceptions that occurred during kernel tuning.
        
        Args:
            exc_type (`type[OutOfResources] | None`):
                The type of exception that was raised, if any.    exc_value (`OutOfResources | None`):
                The exception instance that was raised, if any.    traceback (`TracebackType | None`):
                The traceback object, if any.
        
        Returns:
            `bool`:
                `True` to suppress the exception, `False` to propagate it.
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