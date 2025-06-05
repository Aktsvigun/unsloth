from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    ACT2FN,
    Qwen3MoeSparseMoeBlock,
)

from grouped_gemm.interface import grouped_gemm
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.moe_ops import (
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)

"""
Reference implementation of HF Qwen3 MoE block using grouped gemm.

The Qwen3MoeGroupedGEMMBlock is a reference torch-native implemention.
Qwen3MoeFusedGroupedGEMMBlock is a version using the triton grouped gemm kernel.

NOTE: This is NOT to be used for production as it contains many extra checks and saves all intermediate results for debugging.
"""


@dataclass
class GroupedGEMMResult:
    """
    Data class to store intermediate results of grouped GEMM operations during MoE block computation.
    
    Attributes:
        token_counts_by_expert (`torch.Tensor`):
            Tensor containing the number of tokens assigned to each expert.
        gather_indices (`torch.Tensor`):
            Indices used to gather tokens for each expert.
        topk_weights (`torch.Tensor`):
            Top-k routing weights for each token.
        first_gemm (`torch.Tensor`):
            Output of the first GEMM operation (gate-up projection).
        intermediate (`torch.Tensor`):
            Intermediate result after activation function is applied.
        second_gemm (`torch.Tensor`):
            Output of the second GEMM operation (down projection).
        hidden_states_unpermute (`torch.Tensor`):
            Hidden states after unpermuting from expert order to token order.
        hidden_states (`torch.Tensor`):
            Final output of the MoE block computation.
    """
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    hidden_states: torch.Tensor  # final output


class Qwen3MoeGroupedGEMMBlock(torch.nn.Module):
    """
    Reference implementation of the Qwen3 Mixture-of-Experts (MoE) block using grouped GEMM operations.
    
    This class implements the MoE block using a reference torch-native approach. It processes input through a gating mechanism
    that selects top-k experts, performs grouped GEMM operations for expert computation, and aggregates the results.
    
    Args:
        config (`Qwen3MoeConfig`):
            Configuration object for the Qwen3 MoE model.
        gate (`torch.Tensor`):
            Gate weights of shape (num_experts, hidden_size).
        gate_up_proj (`torch.Tensor`):
            Concatenated gate and up projection weights of shape (num_experts, 2 * moe_intermediate_size, hidden_size).
        down_proj (`torch.Tensor`):
            Down projection weights of shape (num_experts, hidden_size, moe_intermediate_size).
    
    Attributes:
        num_experts (`int`):
            Number of experts in the MoE block.
        top_k (`int`):
            Number of top experts to select per token.    norm_topk_prob (`bool`):
            Whether to normalize the top-k probabilities.
        hidden_size (`int`):
            Size of the input and output hidden states.
        moe_intermediate_size (`int`):
            Size of the intermediate layer in the expert networks.
        gate (`torch.nn.Parameter`):
            Learnable gate weights.
        gate_up_proj (`torch.nn.Parameter`):
            Learnable gate-up projection weights.
        down_proj (`torch.nn.Parameter`):
            Learnable down projection weights.
        act_fn (`Callable`):
            Activation function for the expert networks.
    """
    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        assert gate.shape == (config.num_experts, config.hidden_size)
        assert gate_up_proj.shape == (
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        assert down_proj.shape == (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )

        # gating
        self.gate = torch.nn.Parameter(gate)

        # experts
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj, requires_grad=True)
        self.down_proj = torch.nn.Parameter(down_proj, requires_grad=True)
        self.act_fn = ACT2FN[config.hidden_act]

    @staticmethod
    def extract_hf_weights(moe_block: Qwen3MoeSparseMoeBlock) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts weights from a Hugging Face Qwen3MoeSparseMoeBlock and returns them as tensors.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                Hugging Face MoE block from which to extract weights.
        
        Returns:
            tuple of torch.Tensor: A tuple containing the gate, gate_up_proj, and down_proj weights.
            gate (`torch.Tensor`):
                Gate weights of shape (num_experts, hidden_size).
            gate_up_proj (`torch.Tensor`):
                Concatenated gate and up projection weights of shape (num_experts, 2 * moe_intermediate_size, hidden_size).
            down_proj (`torch.Tensor`):
                Down projection weights of shape (num_experts, hidden_size, moe_intermediate_size).
        """
        config: Qwen3MoeConfig = moe_block.experts[0].config
        num_experts = config.num_experts

        gate = moe_block.gate.weight.data
        gate_proj = torch.stack(
            [moe_block.experts[i].gate_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        up_proj = torch.stack(
            [moe_block.experts[i].up_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        down_proj = torch.stack(
            [moe_block.experts[i].down_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        gate_up_proj = torch.cat([gate_proj, up_proj], dim=1)
        return gate, gate_up_proj, down_proj

    @classmethod
    def from_hf(cls, moe_block: Qwen3MoeSparseMoeBlock) -> Qwen3MoeGroupedGEMMBlock:
        """
        Constructs a Qwen3MoeGroupedGEMMBlock instance from a Hugging Face Qwen3MoeSparseMoeBlock.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                Hugging Face MoE block to convert.
        
        Returns:
            Qwen3MoeGroupedGEMMBlock: A new instance of Qwen3MoeGroupedGEMMBlock with weights extracted from the input block.
        """
        config: Qwen3MoeConfig = moe_block.experts[0].config
        gate, gate_up_proj, down_proj = cls.extract_hf_weights(moe_block)
        return cls(config, gate, gate_up_proj, down_proj)

    def check_weights(self, moe_block: Qwen3MoeSparseMoeBlock) -> None:
        """
        Verifies that the weights in this block match the corresponding weights in the provided Hugging Face MoE block.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                Hugging Face MoE block to compare with.
        
        Raises:
            AssertionError: If any of the weights do not match between the two blocks.
        """
        for i in range(self.num_experts):
            assert self.gate_up_proj[i].equal(
                torch.cat(
                    [
                        moe_block.experts[i].gate_proj.weight.data,
                        moe_block.experts[i].up_proj.weight.data,
                    ],
                    dim=0,
                )
            )
            assert self.down_proj[i].equal(moe_block.experts[i].down_proj.weight.data)

    def act_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the activation function to the gate projection and multiplies it with the up projection.
        
        Args:
            x (`torch.Tensor`):
                Input tensor containing the concatenated gate and up projection outputs.
                Shape: (batch_size * sequence_length, 2 * moe_intermediate_size).
        
        Returns:
            `torch.Tensor`: Result of element-wise multiplication of the activated gate projection and up projection.
                Shape: (batch_size * sequence_length, moe_intermediate_size).
        """
        assert x.shape[-1] == 2 * self.moe_intermediate_size
        gate_proj = x[..., : self.moe_intermediate_size]
        up_proj = x[..., self.moe_intermediate_size :]
        return self.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes router logits and selects top-k experts for each token.
        
        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of shape (batch_size * sequence_length, hidden_dim).
        
        Returns:
            tuple: A tuple containing
                router_logits (`torch.Tensor`):
                    Logits for each expert, shape (batch_size * sequence_length, num_experts).
                routing_weights (`torch.Tensor`):
                    Softmax probabilities of the top-k experts, shape (batch_size * sequence_length, top_k).
                selected_experts (`torch.Tensor`):
                    Indices of the selected top-k experts, shape (batch_size * sequence_length, top_k).
        """
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = torch.nn.functional.linear(hidden_states, self.gate)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts

    def get_token_counts_and_gather_indices(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the number of tokens assigned to each expert and the indices for gathering tokens.
        
        Args:
            selected_experts (`torch.Tensor`):
                Tensor of shape (batch_size * sequence_length, top_k) containing the indices of selected experts.
        
        Returns:
            tuple: A tuple containing
                token_counts_by_expert (`torch.Tensor`):
                    Tensor of shape (num_experts,) containing the number of tokens assigned to each expert.
                gather_indices (`torch.Tensor`):
                    Tensor of shape (total_tokens,) containing the indices for gathering tokens in expert order.
        """
        token_counts_by_expert, gather_indices = get_routing_indices(
            selected_experts, self.num_experts
        )
        assert not token_counts_by_expert.requires_grad
        assert not gather_indices.requires_grad
        return token_counts_by_expert, gather_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k

        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )

        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. Permute tokens from token order to expert order
        hidden_states = permute(hidden_states, gather_indices, self.top_k)
        assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        first_gemm = torch_grouped_gemm(
            X=hidden_states, W=self.gate_up_proj, m_sizes=token_counts_by_expert
        )
        assert first_gemm.shape == (total_tokens, 2 * self.moe_intermediate_size)
        intermediate = self.act_and_mul(first_gemm)
        assert intermediate.shape == (total_tokens, self.moe_intermediate_size)
        second_gemm = torch_grouped_gemm(
            X=intermediate, W=self.down_proj, m_sizes=token_counts_by_expert
        )
        assert second_gemm.shape == (total_tokens, hidden_dim)

        # Post-processing
        # 1. Unpermute from expert order to token order
        hidden_states_unpermute = unpermute(second_gemm, gather_indices)
        assert hidden_states_unpermute.shape == (total_tokens, hidden_dim)

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


class Qwen3MoeFusedGroupedGEMMBlock(Qwen3MoeGroupedGEMMBlock):
    """
    Optimized version of the Qwen3 Mixture-of-Experts (MoE) block using fused grouped GEMM operations.
    
    This class extends Qwen3MoeGroupedGEMMBlock to implement a more efficient version of the MoE block by fusing the permutation
    operations with the grouped GEMM kernels. It allows for autotuning of kernel configurations for optimal performance.
    
    Args:
        config (`Qwen3MoeConfig`):
            Configuration object for the Qwen3 MoE model.
        gate (`torch.Tensor`):
            Gate weights of shape (num_experts, hidden_size).
        gate_up_proj (`torch.Tensor`):
            Concatenated gate and up projection weights of shape (num_experts, 2 * moe_intermediate_size, hidden_size).
        down_proj (`torch.Tensor`):
            Down projection weights of shape (num_experts, hidden_size, moe_intermediate_size).
        permute_x (`bool`, optional, default=True):
            Whether to permute the input tensor for the first GEMM operation.
        permute_y (`bool`, optional, default=True):
            Whether to permute the output tensor for the second GEMM operation.
        autotune (`bool`, optional, default=True):
            Whether to autotune the kernel configurations.
        kernel_config_fwd (`KernelConfigForward`, optional):
            Forward kernel configuration. Required if autotune is False.
        kernel_config_bwd_dW (`KernelConfigBackward_dW`, optional):
            Backward kernel configuration for weight gradients. Required if autotune is False.
        kernel_config_bwd_dX (`KernelConfigBackward_dX`, optional):
            Backward kernel configuration for input gradients. Required if autotune is False.
        dW_only (`bool`, optional, default=False):
            Whether to compute only weight gradients.
        dX_only (`bool`, optional, default=False):
            Whether to compute only input gradients.
    """
    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        permute_x: bool                               = True,
        permute_y: bool                               = True,
        autotune: bool                                = True,
        kernel_config_fwd: KernelConfigForward        = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool                                 = False,
        dX_only: bool                                 = False,
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
        self.dW_only = dW_only
        self.dX_only = dX_only

    @classmethod
    def from_hf(
        cls,
        moe_block: Qwen3MoeSparseMoeBlock,
        permute_x: bool                               = True,
        permute_y: bool                               = True,
        autotune: bool                                = True,
        kernel_config_fwd: KernelConfigForward        = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool                                 = False,
        dX_only: bool                                 = False,
    ) -> Qwen3MoeFusedGroupedGEMMBlock:
        """
        Constructs a Qwen3MoeFusedGroupedGEMMBlock instance from a Hugging Face Qwen3MoeSparseMoeBlock.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                Hugging Face MoE block to convert.
            permute_x (`bool`, optional, default=True):
                Whether to permute the input tensor for the first GEMM operation.
            permute_y (`bool`, optional, default=True):
                Whether to permute the output tensor for the second GEMM operation.
            autotune (`bool`, optional, default=True):
                Whether to autotune the kernel configurations.
            kernel_config_fwd (`KernelConfigForward`, optional):
                Forward kernel configuration. Required if autotune is False.
            kernel_config_bwd_dW (`KernelConfigBackward_dW`, optional):
                Backward kernel configuration for weight gradients. Required if autotune is False.
            kernel_config_bwd_dX (`KernelConfigBackward_dX`, optional):
                Backward kernel configuration for input gradients. Required if autotune is False.
            dW_only (`bool`, optional, default=False):
                Whether to compute only weight gradients.
            dX_only (`bool`, optional, default=False):
                Whether to compute only input gradients.
        
        Returns:
            Qwen3MoeFusedGroupedGEMMBlock: A new instance of Qwen3MoeFusedGroupedGEMMBlock with weights extracted from the input block.
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
            dW_only=dW_only,
            dX_only=dX_only,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MoE block using fused grouped GEMM operations.
        
        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of shape (batch_size, sequence_length, hidden_dim).
        
        Returns:
            tuple: A tuple containing
                hidden_states (`torch.Tensor`):
                    Output tensor of shape (batch_size, sequence_length, hidden_dim).
                router_logits (`torch.Tensor`):
                    Logits for each expert, shape (batch_size * sequence_length, num_experts).
        
        This method implements the MoE block computation using optimized grouped GEMM kernels. It performs the following steps:
        1. Reshapes the input tensor to 2D.
        2. Computes router logits and selects top-k experts.
        3. Computes token counts and gather indices for expert computation.
        4. Performs the first grouped GEMM operation (gate-up projection) with optional permutation fusion.
        5. Applies the activation function and performs the second grouped GEMM operation (down projection) with optional permutation fusion.
        6. Unpermutes the output tensor if needed.
        7. Aggregates the results by applying top-k weights and reshaping back to the original shape.
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
        # Start expert computation
        hidden_states = grouped_gemm(
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
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )
        hidden_states = self.act_and_mul(hidden_states)
        hidden_states = grouped_gemm(
            X=hidden_states,
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
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )

        # Post-processing
        # 1. Unpermute from expert order to token order
        if not self.permute_y:
            hidden_states = unpermute(hidden_states, gather_indices)

        # 2. Merge topk weights
        hidden_states = (
            hidden_states.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )
        hidden_states = hidden_states.sum(dim=1)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits
