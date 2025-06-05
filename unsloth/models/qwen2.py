from typing import Type, Optional, T, Any
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

from .llama import *
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
)
# For Pytorch 2.1.1
try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2SdpaAttention,
        Qwen2FlashAttention2,
    )
except:
    Qwen2SdpaAttention   = Qwen2Attention
    Qwen2FlashAttention2 = Qwen2Attention
pass


class FastQwen2Model(FastLlamaModel):
    """
    A class for the FastQwen2 model, which is a modified version of the Qwen2 model optimized for performance.
    
    This class provides methods to patch the model for improved efficiency and to load the model from a pretrained state.
    
    Methods:
        `pre_patch()`: Applies patches to the model components to optimize performance.
        `from_pretrained()`: Loads a pretrained FastQwen2 model with specified configurations.
    """

    @staticmethod
    def pre_patch() -> None:
        """
        Applies necessary patches to the Qwen2 model components to optimize performance.
        
        This method modifies the forward functions of attention and decoder layers to use fast implementations.
        It also patches the rotary embeddings and prepares the model for efficient inference and training.
        
        Returns:
            None
        """
        init_name, function = patch_linear_scaling(
            model_name         = "qwen2",
            rope_module        = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module   = Qwen2Attention,
        )
        if init_name is not None:
            exec(function, globals())
            Qwen2Attention.__init__  = eval(init_name)
        pass
        Qwen2Attention      .forward = LlamaAttention_fast_forward
        Qwen2SdpaAttention  .forward = LlamaAttention_fast_forward
        Qwen2FlashAttention2.forward = LlamaAttention_fast_forward
        Qwen2DecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        Qwen2Model          .forward = LlamaModel_fast_forward
        Qwen2ForCausalLM    .forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        fix_prepare_inputs_for_generation(Qwen2ForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.qwen2.modeling_qwen2
        transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding = LlamaRotaryEmbedding
        return
    pass


    @staticmethod
    def from_pretrained(
        model_name: str                               = "Qwen/Qwen2-7B",
        max_seq_length: int                           = 4096,
        dtype: Optional[torch.dtype]                  = None,
        load_in_4bit: bool                            = True,
        token: Optional[str]                          = None,
        device_map: str | dict                        = "sequential",
        rope_scaling: Optional[Any]                   = None, # Qwen2 does not support RoPE scaling
        fix_tokenizer: bool                           = True,
        model_patcher: Optional[Type[FastQwen2Model]] = None,
        tokenizer_name: Optional[str]                 = None,
        trust_remote_code: bool                       = False,
        **kwargs,
    ) -> FastQwen2Model:
        """
        Loads a pretrained FastQwen2 model with specified configurations.
        
        Args:
            model_name (`str`, defaults to "Qwen/Qwen2-7B"):
                Name of the pretrained model to load.
            max_seq_length (`int`, defaults to 4096):
                Maximum sequence length for the model.
            dtype (`torch.dtype`, optional):
                Data type for the model weights.
            load_in_4bit (`bool`, defaults to True):
                Whether to load the model in 4-bit precision.
            token (`str`, optional):
                Authentication token for accessing private models.
            device_map (`str` or `dict`, defaults to "sequential"):
                Device map for distributing the model across devices.
            rope_scaling (`Any`, optional):
                RoPE scaling configuration (Qwen2 does not support RoPE scaling).
            fix_tokenizer (`bool`, defaults to True):
                Whether to fix the tokenizer.
            model_patcher (`Type[FastQwen2Model]`, optional):
                Model patcher class to use.
            tokenizer_name (`str`, optional):
                Name of the tokenizer to use.
            trust_remote_code (`bool`, defaults to False):
                Whether to trust remote code when loading the model.
        
        Returns:
            `FastQwen2Model`: The loaded and configured FastQwen2 model.
        """
        return FastLlamaModel.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = dtype,
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = FastQwen2Model,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
    pass
pass