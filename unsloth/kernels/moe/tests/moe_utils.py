from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from huggingface_hub import HfApi
from huggingface_hub.utils import _safetensors
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from grouped_gemm.interface import grouped_gemm
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.layers.qwen3_moe import (
    GroupedGEMMResult,
    Qwen3MoeGroupedGEMMBlock,
)
from grouped_gemm.reference.moe_ops import permute, unpermute


def rebind_experts_to_shared_buffer(
    moe_block: Qwen3MoeSparseMoeBlock, config: Qwen3MoeConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rebinds the weights of each expert in the MoE block to shared buffers.
    
    Args:
        moe_block (`Qwen3MoeSparseMoeBlock`):
            The MoE block containing the experts.
        config (`Qwen3MoeConfig`):
            The configuration object for the MoE model.
    
    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:
            The shared buffers for the up, gate, and down projections of the experts.
    """
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    interm_size = config.moe_intermediate_size
    device = moe_block.experts[0].down_proj.weight.device
    dtype = moe_block.experts[0].down_proj.weight.dtype

    buffer_up = torch.empty(
        num_experts, interm_size, hidden_size, device=device, dtype=dtype
    )
    buffer_gate = torch.empty(
        num_experts, interm_size, hidden_size, device=device, dtype=dtype
    )
    buffer_down = torch.empty(
        num_experts, hidden_size, interm_size, device=device, dtype=dtype
    )

    # Step 2: Copy existing expert weights into buffers
    for i, expert in enumerate(moe_block.experts):
        buffer_up[i].copy_(expert.up_proj.weight.data)
        buffer_gate[i].copy_(expert.gate_proj.weight.data)
        buffer_down[i].copy_(expert.down_proj.weight.data)

    # Step 3: Rebind expert weights to views in shared buffer
    for i, expert in enumerate(moe_block.experts):
        expert.up_proj.weight = torch.nn.Parameter(buffer_up[i])
        expert.gate_proj.weight = torch.nn.Parameter(buffer_gate[i])
        expert.down_proj.weight = torch.nn.Parameter(buffer_down[i])

    return buffer_up, buffer_gate, buffer_down


def get_expert_metadata(model_id: str) -> _safetensors.SafetensorsRepoMetadata:
    """
    Retrieves metadata for expert models from Hugging Face.
    
    Args:
        model_id (`str`):
            The ID of the model on Hugging Face.
    
    Returns:
        `_safetensors.SafetensorsRepoMetadata`:
            The metadata of the model, including file information.
    """
    api = HfApi()
    metadata: _safetensors.SafetensorsRepoMetadata = api.get_safetensors_metadata(
        model_id
    )
    return metadata.files_metadata


def clone_experts(
    moe_block: Qwen3MoeSparseMoeBlock, config: Qwen3MoeConfig, copy: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Clones the weights of each expert in the MoE block into separate tensors.
    
    Args:
        moe_block (`Qwen3MoeSparseMoeBlock`):
            The MoE block containing the experts.
        config (`Qwen3MoeConfig`):
            The configuration object for the MoE model.    copy (`bool`, optional, defaults to `True`):
            Whether to create a copy of the weights.
    
    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:
            The cloned weights for the gate, up, and down projections of the experts.
    """
    down_projs = torch.empty(
        config.num_experts, config.hidden_size, config.moe_intermediate_size
    )
    up_projs = torch.empty(
        config.num_experts, config.moe_intermediate_size, config.hidden_size
    )
    gate_projs = torch.empty(
        config.num_experts, config.moe_intermediate_size, config.hidden_size
    )
    for expert_idx, expert in enumerate(moe_block.experts):
        down_projs[expert_idx].copy_(expert.down_proj.weight.data)
        up_projs[expert_idx].copy_(expert.up_proj.weight.data)
        gate_projs[expert_idx].copy_(expert.gate_proj.weight.data)
    return gate_projs, up_projs, down_projs


@dataclass
class ForwardResult:
    """
    Data class to store the results of the forward pass in a MoE block.
    
    Args:
        output (`torch.Tensor`):
            The output tensor from the forward pass.
        router_logits (`torch.Tensor`):
            The logits from the router in the MoE block.
        X (`torch.Tensor`):
            The input tensor to the forward pass.
        grouped_gemm_result (`GroupedGEMMResult`, optional):
            The intermediate results from the grouped GEMM operations, if applicable.
    """
    output: torch.Tensor
    router_logits: torch.Tensor
    X: torch.Tensor
    # When using grouped gemm MoE implementation to additional debugging / checking of intermediate results
    grouped_gemm_result: GroupedGEMMResult = None



@dataclass
class BackwardResult:
    """
    Data class to store the results of the backward pass in a MoE block.
    
    Args:
        X_grad (`torch.Tensor`):
            The gradient of the input tensor.
        gate_grad (`torch.Tensor`):
            The gradient of the gate weights.
        gate_proj_grad (`torch.Tensor`):
            The gradient of the gate projection weights.
        up_proj_grad (`torch.Tensor`):
            The gradient of the up projection weights.
        down_proj_grad (`torch.Tensor`):
            The gradient of the down projection weights.
    """
    X_grad: torch.Tensor
    gate_grad: torch.Tensor
    gate_proj_grad: torch.Tensor
    up_proj_grad: torch.Tensor
    down_proj_grad: torch.Tensor


def check_down_proj_grad(
    moe_block: Qwen3MoeSparseMoeBlock,
    grouped_gemm_block: Qwen3MoeGroupedGEMMBlock,
    atol: float,
    rtol: float,
) -> None:
    """
    Checks the gradients of the down projection weights between the reference and test models.
    
    Args:
        moe_block (`Qwen3MoeSparseMoeBlock`):
            The reference MoE block.
        grouped_gemm_block (`Qwen3MoeGroupedGEMMBlock`):
            The test MoE block using grouped GEMM.
        atol (`float`):
            The absolute tolerance for gradient comparison.
        rtol (`float`):
            The relative tolerance for gradient comparison.
    
    Raises:
        `AssertionError`: If the gradients do not match within the specified tolerances.
    """
    for i, expert in enumerate(moe_block.experts):
        ref_grad = expert.down_proj.weight.grad
        assert ref_grad is not None
        test_grad = grouped_gemm_block.down_proj.grad[i]
        assert test_grad is not None
        diff = (ref_grad - test_grad).abs().max()
        if not torch.allclose(ref_grad, test_grad, atol=atol, rtol=rtol):
            print(f"expert {i} down_proj_grad_diff: {diff.detach().cpu().item():.6f}")


def check_gate_up_proj_grad(
    moe_block: Qwen3MoeSparseMoeBlock,
    grouped_gemm_block: Qwen3MoeGroupedGEMMBlock,
    atol: float,
    rtol: float,
) -> None:
    """
    Checks the gradients of the gate and up projection weights between the reference and test models.
    
    Args:
        moe_block (`Qwen3MoeSparseMoeBlock`):
            The reference MoE block.
        grouped_gemm_block (`Qwen3MoeGroupedGEMMBlock`):
            The test MoE block using grouped GEMM.
        moe_intermediate_size (`int`):
            The size of the intermediate layer in the MoE block.
        atol (`float`):
            The absolute tolerance for gradient comparison.
        rtol (`float`):
            The relative tolerance for gradient comparison.
    
    Raises:
        `AssertionError`: If the gradients do not match within the specified tolerances.
    """
    moe_intermediate_size = grouped_gemm_block.moe_intermediate_size
    for i, expert in enumerate(moe_block.experts):
        ref_gate_proj_grad = expert.gate_proj.weight.grad
        ref_up_proj_grad = expert.up_proj.weight.grad
        assert ref_gate_proj_grad is not None
        assert ref_up_proj_grad is not None

        # Extract gradients
        test_gate_proj_grad = grouped_gemm_block.gate_up_proj.grad[
            i, :moe_intermediate_size
        ]
        test_up_proj_grad = grouped_gemm_block.gate_up_proj.grad[
            i, moe_intermediate_size:
        ]
        assert test_gate_proj_grad is not None
        assert test_up_proj_grad is not None

        # Sanity check shapes
        assert ref_gate_proj_grad.shape == test_gate_proj_grad.shape, (
            f"{ref_gate_proj_grad.shape} != {test_gate_proj_grad.shape}"
        )
        assert ref_up_proj_grad.shape == test_up_proj_grad.shape, (
            f"{ref_up_proj_grad.shape} != {test_up_proj_grad.shape}"
        )

        # Check gradients
        diff = (ref_gate_proj_grad - test_gate_proj_grad).abs().max()
        if not torch.allclose(
            ref_gate_proj_grad, test_gate_proj_grad, atol=atol, rtol=rtol
        ):
            print(f"expert {i} gate_proj_grad_diff: {diff.detach().cpu().item():.6f}")
        diff = (ref_up_proj_grad - test_up_proj_grad).abs().max()
        if not torch.allclose(
            ref_up_proj_grad, test_up_proj_grad, atol=atol, rtol=rtol
        ):
            print(f"expert {i} up_proj_grad_diff: {diff.detach().cpu().item():.6f}")


def check_gate_grad(
    moe_block: Qwen3MoeSparseMoeBlock,
    grouped_gemm_block: Qwen3MoeGroupedGEMMBlock,
    atol: float,
    rtol: float,
) -> None:
    """
    Checks the gradients of the gate weights between the reference and test models.
    
    Args:
        moe_block (`Qwen3MoeSparseMoeBlock`):
            The reference MoE block.
        grouped_gemm_block (`Qwen3MoeGroupedGEMMBlock`):
            The test MoE block using grouped GEMM.
        atol (`float`):
            The absolute tolerance for gradient comparison.
        rtol (`float`):
            The relative tolerance for gradient comparison.
    
    Raises:
        `AssertionError`: If the gradients do not match within the specified tolerances.
    """
    ref_grad = moe_block.gate.weight.grad
    assert ref_grad is not None
    test_grad = grouped_gemm_block.gate.grad
    assert test_grad is not None
    diff = (ref_grad - test_grad).abs().max()
    if not torch.allclose(ref_grad, test_grad, atol=atol, rtol=rtol):
        print(f"gate_grad_diff: {diff.detach().cpu().item():.6f}")


def check_wgrad(
    moe_block: Qwen3MoeSparseMoeBlock,
    grouped_gemm_block: Qwen3MoeGroupedGEMMBlock,
    atol: float,
    rtol: float,
) -> None:
    """
    Checks the gradients of all weights in the MoE block between the reference and test models.
    
    Args:
        moe_block (`Qwen3MoeSparseMoeBlock`):
            The reference MoE block.
        grouped_gemm_block (`Qwen3MoeGroupedGEMMBlock`):
            The test MoE block using grouped GEMM.
        atol (`float`):
            The absolute tolerance for gradient comparison.
        rtol (`float`):
            The relative tolerance for gradient comparison.
    
    Raises:
        `AssertionError`: If any gradients do not match within the specified tolerances.
    """
    check_down_proj_grad(moe_block, grouped_gemm_block, atol, rtol)
    check_gate_up_proj_grad(moe_block, grouped_gemm_block, atol, rtol)
    check_gate_grad(moe_block, grouped_gemm_block, atol, rtol)


def check_tensor_allclose(
    X_ref: torch.Tensor,
    X_test: torch.Tensor,
    atol: float,
    rtol: float,
    name: str,
    verbose: bool = False,
) -> None:
    """
    Checks if two tensors are close to each other within the specified tolerances.
    
    Args:
        X_ref (`torch.Tensor`):
            The reference tensor.
        X_test (`torch.Tensor`):
            The test tensor.
        atol (`float`):
            The absolute tolerance for comparison.
        rtol (`float`):
            The relative tolerance for comparison.    name (`str`):
            The name of the tensor for logging purposes.    verbose (`bool`, optional, defaults to `False`):
            Whether to print the difference if the tensors are not close.
    
    Raises:
        `AssertionError`: If the tensors are not close within the specified tolerances.
    """
    diff = (X_ref - X_test).abs().max()
    if verbose:
        print(f"{name} diff: {diff.detach().cpu().item():.6f}")
    assert torch.allclose(X_ref, X_test, atol=atol, rtol=rtol), (
        f"{name} diff: {diff.detach().cpu().item():.6f}"
    )


def check_expert_grads(
    ref_result: BackwardResult,
    test_result: BackwardResult,
    atol: float,
    rtol: float,
    verbose: bool = False,
) -> None:
    """
    Checks the gradients of the expert weights between the reference and test models.
    
    Args:
        ref_result (`BackwardResult`):
            The reference backward result containing gradients.
    test_result (`BackwardResult`):
            The test backward result containing gradients.
        atol (`float`):
            The absolute tolerance for gradient comparison.
        rtol (`float`):
            The relative tolerance for gradient comparison.    verbose (`bool`, optional, defaults to `False`):
            Whether to print the differences if the gradients are not close.
    
    Raises:
        `AssertionError`: If any expert gradients do not match within the specified tolerances.
    """
    fields_to_check = [f.name for f in fields(BackwardResult) if "proj" in f.name]
    assert len(fields_to_check) == 3

    for field in fields_to_check:
        ref_grads = getattr(ref_result, field)
        test_grads = getattr(test_result, field)
        assert ref_grads.shape == test_grads.shape, (
            f"{field}: {ref_grads.shape} != {test_grads.shape}"
        )

        # Test each expert
        for i in range(ref_grads.shape[0]):
            ref_grad = ref_grads[i]
            test_grad = test_grads[i]
            diff = (ref_grad - test_grad).abs().max()
            assert torch.allclose(ref_grad, test_grad, atol=atol, rtol=rtol), (
                f"{field}[{i}] diff: {diff.detach().cpu().item():.6f}"
            )

        # Test all experts
        diff = (ref_grads - test_grads).abs().max()
        if verbose:
            print(f"{field} diff: {diff.detach().cpu().item():.6f}")
        assert torch.allclose(ref_grads, test_grads, atol=atol, rtol=rtol), (
            f"{field} diff: {diff.detach().cpu().item():.6f}"
        )


def check_grads(
    ref_result: BackwardResult,
    test_result: BackwardResult,
    atol: float,
    rtol: float,
    verbose: bool = False,
) -> None:
    """
    Checks the gradients of all components in the backward pass between the reference and test models.
    
    Args:
        ref_result (`BackwardResult`):
            The reference backward result containing gradients.
    test_result (`BackwardResult`):
            The test backward result containing gradients.
        atol (`float`):
            The absolute tolerance for gradient comparison.
        rtol (`float`):
            The relative tolerance for gradient comparison.    verbose (`bool`, optional, defaults to `False`):
            Whether to print the differences if the gradients are not close.
    
    Raises:
        `AssertionError`: If any gradients do not match within the specified tolerances.
    """
    check_tensor_allclose(
        ref_result.X_grad, test_result.X_grad, atol, rtol, "X.grad", verbose
    )
    check_tensor_allclose(
        ref_result.gate_grad, test_result.gate_grad, atol, rtol, "gate.grad", verbose
    )
    check_expert_grads(ref_result, test_result, atol, rtol, verbose)


def check_fwd(
    ref_result: ForwardResult,
    test_result: ForwardResult,
    atol: float,
    rtol: float,
    verbose: bool = False,
) -> None:
    """
    Checks the forward pass results between the reference and test models.
    
    Args:
        ref_result (`ForwardResult`):
            The reference forward result containing outputs.
    test_result (`ForwardResult`):
            The test forward result containing outputs.
        atol (`float`):
            The absolute tolerance for output comparison.
        rtol (`float`):
            The relative tolerance for output comparison.    verbose (`bool`, optional, defaults to `False`):
            Whether to print the differences if the outputs are not close.
    
    Raises:
        `AssertionError`: If any outputs do not match within the specified tolerances.
    """
    # First check hidden states (output)
    ref_output = ref_result.output
    test_output = test_result.output
    diff = (ref_output - test_output).abs().max()
    if verbose:
        print(f"output diff: {diff.detach().cpu().item():.6f}")
    assert torch.allclose(ref_output, test_output, atol=atol, rtol=rtol), (
        f"output diff: {diff.detach().cpu().item():.6f}"
    )

    # Check router logits
    ref_router_logits = ref_result.router_logits
    test_router_logits = test_result.router_logits
    diff = (ref_router_logits - test_router_logits).abs().max()
    if verbose:
        print(f"router_logits diff: {diff.detach().cpu().item():.6f}")
    assert torch.allclose(
        ref_router_logits, test_router_logits, atol=atol, rtol=rtol
    ), f"router_logits diff: {diff.detach().cpu().item():.6f}"


def check_grouped_gemm_results(
    grouped_result: GroupedGEMMResult,
    fused_result: GroupedGEMMResult,
    permute_y: bool,
    atol: float,
    rtol: float,
    verbose: bool = False,
) -> None:
    """
    Checks the results of grouped GEMM operations between the reference and test models.
    
    Args:
        grouped_result (`GroupedGEMMResult`):
            The reference grouped GEMM result.
    fused_result (`GroupedGEMMResult`):
            The test grouped GEMM result.    permute_y (`bool`):
            Whether the output should be permuted.
        atol (`float`):
            The absolute tolerance for comparison.
        rtol (`float`):
            The relative tolerance for comparison.    verbose (`bool`, optional, defaults to `False`):
            Whether to print the differences if the results are not close.
    
    Raises:
        `AssertionError`: If any results do not match within the specified tolerances.
    """
    for field in fields(GroupedGEMMResult):
        ref_value = getattr(grouped_result, field.name)
        test_value = getattr(fused_result, field.name)
        diff = (ref_value - test_value).abs().max()

        # second_gemm in torch grouped gemm is not yet unpermuted so comparing the fused unpermuted second_gemm will result in error
        # instead the hidden_states_unpermute should match since hidden_states_unpermute for the fused result is the same as second_gemm
        if field.name == "second_gemm" and permute_y:
            continue

        if verbose:
            print(f"{field.name} diff: {diff.detach().cpu().item():.6f}")

        assert torch.allclose(ref_value, test_value, atol=atol, rtol=rtol), (
            f"{field.name} diff: {diff.detach().cpu().item():.6f}"
        )


def run_forward(model: nn.Module, X: torch.Tensor, is_grouped_gemm: bool = False) -> ForwardResult:
    """
    Runs the forward pass of the model and returns the results.
    
    Args:
        model (`nn.Module`):
            The model to run the forward pass on.    X (`torch.Tensor`):
            The input tensor.    is_grouped_gemm (`bool`, optional, defaults to `False`):
            Whether the model uses grouped GEMM operations.
    
    Returns:
        `ForwardResult`:
            The result of the forward pass, including output, router logits, and input tensor.
    """
    X = X.detach().clone().requires_grad_(True)
    output, router_logits = model(X)
    if is_grouped_gemm:
        result = ForwardResult(
            output=output.hidden_states,
            router_logits=router_logits,
            X=X,
            grouped_gemm_result=output,
        )
    else:
        result = ForwardResult(output=output, router_logits=router_logits, X=X)
    return result


def run_backward(
    model: nn.Module, grad_output: torch.Tensor, output: torch.Tensor, X: torch.Tensor
) -> BackwardResult:
    """
    Runs the backward pass of the model and returns the gradients.
    
    Args:
        model (`nn.Module`):
            The model to run the backward pass on.    grad_output (`torch.Tensor`):
            The gradient of the output tensor.    output (`torch.Tensor`):
            The output tensor from the forward pass.    X (`torch.Tensor`):
            The input tensor.
    
    Returns:
        `BackwardResult`:
            The result of the backward pass, including gradients of the input and model parameters.
    """
    output.backward(grad_output)
    assert X.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} grad is None"
    if isinstance(model, Qwen3MoeSparseMoeBlock):
        gate_grad = model.gate.weight.grad
        gate_proj_grad = torch.stack(
            [expert.gate_proj.weight.grad for expert in model.experts]
        )
        up_proj_grad = torch.stack(
            [expert.up_proj.weight.grad for expert in model.experts]
        )
        down_proj_grad = torch.stack(
            [expert.down_proj.weight.grad for expert in model.experts]
        )
    elif isinstance(model, Qwen3MoeGroupedGEMMBlock):
        gate_grad = model.gate.grad
        gate_proj_grad, up_proj_grad = model.gate_up_proj.grad.chunk(2, dim=1)
        down_proj_grad = model.down_proj.grad
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    return BackwardResult(
        X_grad=X.grad,
        gate_grad=gate_grad,
        gate_proj_grad=gate_proj_grad,
        up_proj_grad=up_proj_grad,
        down_proj_grad=down_proj_grad,
    )


class Qwen3MoeFusedGroupedGEMMBlock(Qwen3MoeGroupedGEMMBlock):
    """
    Reference implementation of MoE block using grouped gemm.

    This is the same as the Qwen3MoeGroupedGEMMBlock but with triton grouped gemm in place of torch-native grouped gemm implementation.

    NOTE: This is NOT to be used for production as it contains many extra checks and saves all intermediate results for debugging.
    See grouped_gemm/reference/moe_block.py for a cleaner implementation.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        permute_x: bool                               = False,
        permute_y: bool                               = False,
        autotune: bool                                = True,
        kernel_config_fwd: KernelConfigForward        = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    ):
        super().__init__(config, gate, gate_up_proj, down_proj)
        self.permute_x = permute_x
        self.permute_y = permute_y
        self.autotune = autotune
        if not autotune:
            assert (
                kernel_config_fwd is not None
                and kernel_config_bwd_dW is not None
                and kernel_config_bwd_dX is not None
            ), "Kernel configs must be provided if autotune is False"
        self.kernel_config_fwd = kernel_config_fwd
        self.kernel_config_bwd_dW = kernel_config_bwd_dW
        self.kernel_config_bwd_dX = kernel_config_bwd_dX

    @classmethod
    def from_hf(
        cls,
        moe_block: Qwen3MoeSparseMoeBlock,
        permute_x: bool                               = False,
        permute_y: bool                               = False,
        autotune: bool                                = True,
        kernel_config_fwd: KernelConfigForward        = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    ) -> Qwen3MoeFusedGroupedGEMMBlock:
        """
        Creates a `Qwen3MoeFusedGroupedGEMMBlock` from a Hugging Face `Qwen3MoeSparseMoeBlock`.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                The Hugging Face MoE block to convert.    permute_x (`bool`, optional, defaults to `False`):
                Whether to permute the input tensor.    permute_y (`bool`, optional, defaults to `False`):
                Whether to permute the output tensor.    autotune (`bool`, optional, defaults to `True`):
                Whether to automatically tune the kernel configurations.    kernel_config_fwd (`KernelConfigForward`, optional):
                The kernel configuration for the forward pass.    kernel_config_bwd_dW (`KernelConfigBackward_dW`, optional):
                The kernel configuration for the backward pass of the weight gradients.    kernel_config_bwd_dX (`KernelConfigBackward_dX`, optional):
                The kernel configuration for the backward pass of the input gradients.
        
        Returns:
            `Qwen3MoeFusedGroupedGEMMBlock`:
                The converted MoE block using grouped GEMM operations.
        """
        config: Qwen3MoeConfig = moe_block.experts[0].config
        gate, gate_up_proj, down_proj = Qwen3MoeGroupedGEMMBlock.extract_hf_weights(
            moe_block
        )
        return cls(
            config,
            gate,
            gate_up_proj,
            down_proj,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
        )

    def forward(self, hidden_states: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Performs the forward pass of the MoE block using grouped GEMM operations.
        
        Args:
            hidden_states (`torch.Tensor`):
                The input tensor to the MoE block.    debug (`bool`, optional, defaults to `False`):
                Whether to enable debug mode, which may include additional checks or logging.
        
        Returns:
            `torch.Tensor`:
                The output tensor from the MoE block, along with the router logits.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k

        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )
        # Pre-processing
        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. permute_x -> permutation will be fused in prologue of first grouped gemm
        if not self.permute_x:
            hidden_states = permute(hidden_states, gather_indices, self.top_k)
            assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        first_gemm = grouped_gemm(
            X=hidden_states,
            W=self.gate_up_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=self.permute_x,
            permute_y=False,  # output of first grouped gemm should never be permuted
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=True,
        )
        assert first_gemm.shape == (total_tokens, 2 * self.moe_intermediate_size)
        intermediate = self.act_and_mul(first_gemm)
        assert intermediate.shape == (total_tokens, self.moe_intermediate_size)
        second_gemm = grouped_gemm(
            X=intermediate,
            W=self.down_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=False,
            permute_y=self.permute_y,
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=False,
        )
        assert second_gemm.shape == (total_tokens, hidden_dim)

        # Post-processing
        # 1. Unpermute from expert order to token order
        if not self.permute_y:
            hidden_states_unpermute = unpermute(second_gemm, gather_indices)
            assert hidden_states_unpermute.shape == (total_tokens, hidden_dim)
        else:
            hidden_states_unpermute = second_gemm

        # 2. Merge topk weights
        hidden_states = (
            hidden_states_unpermute.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )
        hidden_states = hidden_states.sum(dim=1)
        assert hidden_states.shape == (num_tokens, hidden_dim)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return GroupedGEMMResult(
            token_counts_by_expert=token_counts_by_expert,
            gather_indices=gather_indices,
            topk_weights=routing_weights,
            first_gemm=first_gemm,
            intermediate=intermediate,
            second_gemm=second_gemm,
            hidden_states_unpermute=hidden_states_unpermute,
            hidden_states=hidden_states,
        ), router_logits
