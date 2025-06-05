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

import torch
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
try:
    from transformers import AutoModelForImageTextToText
    AutoModelForVision2Seq = AutoModelForImageTextToText
except:
    from transformers import AutoModelForVision2Seq
pass
from ..kernels import (
    post_patch_loss_function,
)
from ._utils import __version__
from ._utils import *
from ..save import patch_saving_functions
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from peft import PeftModelForCausalLM
from transformers import set_seed as transformers_set_seed
from unsloth_zoo.peft_utils import (
    get_peft_regex,
    SKIP_QUANTIZATION_MODULES,
    requires_grad_for_gradient_checkpointing,
)
from transformers.models.llama.modeling_llama import logger
from transformers import __version__ as transformers_version
from triton import __version__ as triton_version
from unsloth_zoo.utils import _get_dtype
from unsloth_zoo.patching_utils import patch_model_and_tokenizer
from unsloth_zoo.training_utils import prepare_model_for_training
import types
import functools
import os
import gc
import math
import functools
from typing import Optional, Tuple, List, Union, Dict, Any
import re, inspect, sys
import contextlib
import types
try:
    from huggingface_hub.utils import get_token
except:
    # Old HF Hub versions <= 0.0.25
    from huggingface_hub.utils._token import get_token
pass

__all__ = [
    "FastBaseModel",
]

global NUM_LOGITS_TO_KEEP
NUM_LOGITS_TO_KEEP = dict()
global PROMPT_LOOPKUP
PROMPT_LOOPKUP = dict()

from transformers import GenerationConfig, CompileConfig, HybridCache
_compile_config = CompileConfig(
    fullgraph = False,
    dynamic = None,
    mode = "reduce-overhead",
)
_compile_config.disable = True # Must set manually

from unsloth_zoo.vllm_utils import (
    convert_lora_modules,
    return_lora_modules,
)

def unsloth_base_fast_generate(
    self,
    *args,
    **kwargs,
):
    """
    Enhanced generate method for fast generation with optimized settings and hardware-aware configurations.
    
    Args:
        self: The model instance
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    
    Returns:
        output: The generated output from the model
    
    This method implements various optimizations for generation including:
    - Automatic input detection
    - Batch size handling
    - Model-specific configuration
    - Token type handling
    - VLM (Vision-Language Model) support
    - Data type management
    - LoRA module handling
    - Cache configuration
    - Generation configuration
    - Hardware-specific optimizations
    
    The method also includes error handling and fallback mechanisms for different model architectures.
    """
    if len(args) != 0:
        input_ids = args[0]
    elif "input_ids" in kwargs:
        input_ids = kwargs["input_ids"]
    elif "input" in kwargs:
        input_ids = kwargs["input_ids"]
    elif "input_features" in kwargs:
        input_ids = kwargs["input_features"]
    elif "input_embeds" in kwargs:
        input_ids = kwargs["input_embeds"]
    elif "inputs" in kwargs:
        input_ids = kwargs["inputs"]
    else:
        key = next(iter(kwargs.keys()))
        if type(kwargs["key"]) is not torch.Tensor:
            raise TypeError("Unsloth: You need to pass in input_ids to .generate!")
        input_ids = kwargs[key]
    pass
    assert(type(input_ids) is torch.Tensor)
    bsz = input_ids.shape[0]

    FastBaseModel.for_inference(self)
    dtype = _get_dtype(self.config.torch_dtype)

    # Check if VLM
    is_vlm = any(
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for x in self.config.architectures
    )
    is_vlm = is_vlm or hasattr(self.config, "vision_config")
    arch = self.config.architectures[0]

    # Remove token_type_ids - WRONG for Gemma 3 since bidirectional attention
    if hasattr(self, "generate") and hasattr(self, "forward"):
        # did not combine with below since self might not have model
        keys = inspect.signature(self.forward).parameters.keys()
        if "token_type_ids" not in keys:
            kwargs.pop("token_type_ids", None)
    # kwargs.pop("token_type_ids", None)

    # VLMs do not allow logits_to_keep
    global NUM_LOGITS_TO_KEEP
    if arch not in NUM_LOGITS_TO_KEEP:
        m = self
        # Find which is needed ie
        # num_logits_to_keep or logits_to_keep
        while hasattr(m, "model"):
            if hasattr(m, "forward"):
                keys = inspect.signature(m.forward).parameters.keys()
                if "num_logits_to_keep" in keys:
                    NUM_LOGITS_TO_KEEP[arch] = "num_logits_to_keep"
                    break
                elif "logits_to_keep" in keys:
                    NUM_LOGITS_TO_KEEP[arch] = "logits_to_keep"
                    break
            m = m.model
        pass
        if arch not in NUM_LOGITS_TO_KEEP:
            NUM_LOGITS_TO_KEEP[arch] = None
        pass
    pass
    key = NUM_LOGITS_TO_KEEP[arch]
    if key is not None and key not in kwargs:
        kwargs[key] = 1
    global PROMPT_LOOPKUP
    if arch not in PROMPT_LOOPKUP:
        # Only works for VLMs and not LLMs!
        if is_vlm:
            PROMPT_LOOPKUP[arch] = False
        else:
            PROMPT_LOOPKUP[arch] = True
    if bsz == 1 and PROMPT_LOOPKUP[arch]:
        kwargs["prompt_lookup_num_tokens"] = 3

    # Check pad_token
    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    # Get pixel values for VLMs
    try: kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype)
    except: pass

    # Mixed precision autocast
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
        autocaster = torch.autocast(device_type = "cuda", dtype = torch.float16)
        dtype = torch.float16
    else:
        autocaster = torch.autocast(device_type = "cuda", dtype = dtype)

    # Prepare LoRA
    # state_dict = convert_lora_modules(self, dtype = dtype)

    # Set compile dynamic shapes
    torch._dynamo.mark_static(input_ids, 0)
    torch._dynamo.mark_dynamic(input_ids, 1)
    if "attention_mask" in kwargs:
        torch._dynamo.mark_static(kwargs["attention_mask"], 0)
        torch._dynamo.mark_dynamic(kwargs["attention_mask"], 1)
    if "token_type_ids" in kwargs:
        torch._dynamo.mark_static(kwargs["token_type_ids"], 0)
        torch._dynamo.mark_dynamic(kwargs["token_type_ids"], 1)

    # Fix generation_config
    # Use hybrid if sliding window seen, otherwise try static
    cache_implementation = getattr(self.config, "cache_implementation", None)
    if getattr(self, "_supports_static_cache", True):
        if os.environ.get("UNSLOTH_DISABLE_STATIC_GENERATION", "0") == "0":
            cache_implementation = "static"
        else:
            cache_implementation = None
    else:
        cache_implementation = None
    if cache_implementation is not None:
        swa = getattr(getattr(self.config, "text_config", self.config), "sliding_window", None)
        if swa == 0 or type(swa) is not int:
            cache_implementation = "static"
        else:
            cache_implementation = "hybrid"

    if "generation_config" in kwargs:
        kwargs["generation_config"].cache_implementation = cache_implementation
        if cache_implementation is not None:
            kwargs["generation_config"].compile_config = _compile_config
    else:
        kwargs["cache_implementation"] = cache_implementation
        if cache_implementation is not None:
            kwargs["compile_config"] = _compile_config
    pass

    try:
        with torch.inference_mode(), autocaster:
            output = self._old_generate(*args, **kwargs)
    except:
        PROMPT_LOOPKUP[arch] = False
        kwargs.pop("prompt_lookup_num_tokens", None)
        with torch.inference_mode(), autocaster:
            output = self._old_generate(*args, **kwargs)
    finally:
        pass
        # return_lora_modules(self, state_dict, torch.float32)
    pass

    FastBaseModel.for_training(self)
    return output
pass


class FastBaseModel:
    """
    Base class for fast model operations with optimized loading and training capabilities.
    
    This class provides static methods for model initialization, PEFT model creation,
    and post-processing operations. It includes functionality for:
    - Model loading with quantization support
    - LoRA adapter integration
    - Gradient checkpointing
    - Hardware-specific optimizations
    - Training/inference mode switching
    - Memory management
    - Tokenizer configuration
    
    The class is designed to work with various model architectures and provides
    performance enhancements for both training and inference scenarios.
    """

    @staticmethod
    def from_pretrained(
        model_name: str                              = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length: int                          = 2048,
        dtype: Optional[torch.dtype]                 = None,
        load_in_4bit: bool                           = True,
        load_in_8bit: bool                           = False,
        full_finetuning: bool                        = False,
        token: Optional[str]                         = None,
        device_map: str                              = "sequential",
        trust_remote_code: bool                      = False,
        model_types: Optional[List[str]]             = None,
        tokenizer_name: Optional[str]                = None,
        auto_model: AutoModelForVision2Seq           = AutoModelForVision2Seq,
        use_gradient_checkpointing: Union[bool, str] = "unsloth",
        supports_sdpa: bool                          = True,
        whisper_language: Optional[str]              = None,
        whisper_task: Optional[str]                  = None,
        **kwargs,
    ) -> Tuple[Any, AutoTokenizer]:
        """
        Load a model from a pretrained checkpoint with optimized configuration.
        
        Args:
            model_name (str): Name or path of the pretrained model
            max_seq_length (int): Maximum sequence length for the model
            dtype (torch.dtype, optional): Data type for model weights
            load_in_4bit (bool): Whether to load the model in 4-bit quantization
            load_in_8bit (bool): Whether to load the model in 8-bit quantization
            full_finetuning (bool): Whether to enable full finetuning mode
            token (str, optional): Authentication token for private models
            device_map (str): Device mapping strategy
            trust_remote_code (bool): Whether to trust remote code execution
            model_types (List[str], optional): List of model types
            tokenizer_name (str, optional): Name or path for the tokenizer
            auto_model (AutoModelForVision2Seq): Auto model class for vision models
            use_gradient_checkpointing (Union[bool, str]): Gradient checkpointing configuration
            supports_sdpa (bool): Whether the model supports SDPA
            whisper_language (str, optional): Language for Whisper models
            whisper_task (str, optional): Task for Whisper models
            **kwargs: Additional arguments for model loading
        
        Returns:
            Tuple[Any, AutoTokenizer]: A tuple containing the loaded model and tokenizer
        
        This method handles model loading with various optimizations including:
        - Quantization support
        - Data type configuration
        - Tokenizer setup
        - Model-specific patches
        - Hardware-aware optimizations
        - Memory management
        
        It also provides detailed logging and environment configuration for
        the loaded model.
        """
        if model_types is None:
            raise RuntimeError(
                "Unsloth: Please use FastModel or FastVisionModel and not use FastBaseModel directly!"
            )

        os.environ["UNSLOTH_USE_NEW_MODEL"] = "1"
        if trust_remote_code:
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"\
                "Are you certain you want to do remote code execution?"
            )
        pass
        if token is None: token = get_token()
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        from importlib.metadata import version as importlib_version
        try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
        except: vllm_version = ""

        model_type_arch = model_types[0]
        if model_type_arch == "siglip":
            for model_type_arch in model_types:
                if model_type_arch != "siglip": break

        statistics = \
           f"==((====))==  Unsloth {__version__}: Fast {model_type_arch.title()} patching. Transformers: {transformers_version}.{vllm_version}\n"\
           f"   {chr(92)}{chr(92)}   /|    {gpu_stats.name}. Num GPUs = {torch.cuda.device_count()}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"\
           f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}. Triton: {triton_version}\n"\
           f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
           f' "-____-"     Free license: http://github.com/unslothai/unsloth'
        print(statistics)

        # Warn about fast transfers
        if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
            old_hf_transfer = os.environ["HF_HUB_ENABLE_HF_TRANSFER"]
            if old_hf_transfer in ("False", "false"): old_hf_transfer = "0"
            if old_hf_transfer in ("True",  "true" ): old_hf_transfer = "1"
        else:
            old_hf_transfer = "0"
        if old_hf_transfer == "1":
            print("Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!")
        pass
        if old_hf_transfer != "0": os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        get_statistics() # For debugging - we use a download counter to see if environments are not breaking 

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            if dtype == torch.float16: dtype = torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16
        pass
        assert(dtype in (torch.float16, torch.bfloat16, torch.float32))

        bnb_compute_dtype = dtype
        do_forced_float32 = False
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            print(f"Unsloth: Using float16 precision for {model_type_arch} won't work! Using float32.")
            bnb_compute_dtype = torch.float16
            do_forced_float32 = True
        pass

        # Check for custom data-types
        custom_datatype = None
        correct_dtype = None
        if os.environ.get("UNSLOTH_FORCE_CUSTOM_DTYPE", "") != "":
            custom_datatype = os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"]
            assert custom_datatype.count(";") == 1
            bnb_compute_dtype, custom_datatype = custom_datatype.split(";", 1)
            dtype = torch.float32
            bnb_compute_dtype = eval(bnb_compute_dtype)
            correct_dtype = bnb_compute_dtype
        pass

        # Stop SDPA for some archs like Pixtral / Mistral3
        if not ("attn_implementation" in kwargs):
            kwargs["attn_implementation"] = "sdpa"
        if not supports_sdpa:
            print(f"Unsloth: {model_type_arch.title()} does not support SDPA - switching to eager!")
            del kwargs["attn_implementation"]
        pass

        bnb_config = None
        if full_finetuning and (load_in_4bit or load_in_8bit):
            print("Unsloth: You selected full finetuning support, but 4bit / 8bit is enabled - disabling LoRA / QLoRA.")
            load_in_4bit = False
            load_in_8bit = False
        pass

        if load_in_4bit and load_in_8bit:
            raise RuntimeError("Unsloth: Can only load in 4bit or 8bit, not both!")
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = bnb_compute_dtype,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES.copy(),
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit              = True,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES.copy(),
            )
        elif not load_in_4bit and not load_in_8bit and not full_finetuning:
            print("Unsloth: QLoRA and full finetuning all not selected. Switching to 16bit LoRA.")
        pass

        if full_finetuning:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"
            if dtype == torch.bfloat16:
                print("Unsloth: Using bfloat16 full finetuning which cuts memory usage by 50%.")
            else:
                print("Unsloth: Float16 full finetuning uses more memory since we upcast weights to float32.")
        else:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "0"
        pass

        # Cannot be None, since HF now checks for the config
        if load_in_4bit: kwargs["quantization_config"] = bnb_config

        # Check if using forced float32 - we load it in bfloat16, then cast to float16!
        torch_dtype = dtype
        if do_forced_float32: torch_dtype = torch.bfloat16

        model = auto_model.from_pretrained(
            model_name,
            device_map              = device_map,
            torch_dtype             = torch_dtype,
            # quantization_config   = bnb_config,
            token                   = token,
            trust_remote_code       = trust_remote_code,
            # attn_implementation   = attn_implementation,
            **kwargs,
        )
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Edit data-types
        if custom_datatype is not None:
            for name, module in model.named_modules():
                exec(custom_datatype)
        pass

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        is_vlm = (auto_model is AutoModelForVision2Seq)
        is_whisper = (whisper_language is not None and whisper_task is not None)
        auto_processor = AutoProcessor if (is_vlm or is_whisper) else AutoTokenizer
        if (whisper_language and whisper_task) or auto_model.__name__.endswith("ForConditionalGeneration"):
           tokenizer = auto_processor.from_pretrained(
                tokenizer_name,
                padding_side = "right",
                token        = token,
                language     = whisper_language,
                task         = whisper_task,
            )
        else:
            tokenizer = auto_processor.from_pretrained(
                tokenizer_name,
                padding_side = "right",
                token        = token,
            )
        if hasattr(tokenizer, "tokenizer"):
            __tokenizer = tokenizer.tokenizer
            # Add padding side as well
            __tokenizer.padding_side = "right"
            # Check bos, eos, pad tokens
            if hasattr(__tokenizer, "bos_token"):
                tokenizer.bos_token    = __tokenizer.bos_token
                tokenizer.bos_token_id = __tokenizer.bos_token_id
            if hasattr(__tokenizer, "eos_token"):
                tokenizer.eos_token    = __tokenizer.eos_token
                tokenizer.eos_token_id = __tokenizer.eos_token_id
            if hasattr(__tokenizer, "pad_token"):
                tokenizer.pad_token    = __tokenizer.pad_token
                tokenizer.pad_token_id = __tokenizer.pad_token_id
        pass
        # Fix other stuff like BnB compute data types
        model, tokenizer = patch_model_and_tokenizer(
            model,
            tokenizer,
            downcast_rope = False,
            fix_embeddings = False,
            do_forced_float32 = do_forced_float32,
            correct_dtype = correct_dtype,
        )
        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = post_patch_loss_function(model)

        # Log Unsloth version for future fastpaths for inference
        if hasattr(model, "config"):
            model.config.update({"unsloth_version" : __version__})
        pass
        patch_saving_functions(model, vision = True)
        patch_saving_functions(tokenizer, vision = True)

        # Fix gradient accumulation
        from transformers.trainer import Trainer
        patch_gradient_accumulation_fix(Trainer)

        # Save tokenizer for inference purposes
        tokenizer.padding_side = "left" # Force inference
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.padding_side = "left" # Force inference
        m = model
        while hasattr(m, "model"):
            m.max_seq_length = max_seq_length
            m._saved_temp_tokenizer = tokenizer
            # Also set is_loaded_in_8bit to disable incorrect DDP
            m.is_loaded_in_8bit = True if not full_finetuning else False
            m = m.model
        pass
        m.max_seq_length = max_seq_length
        m._saved_temp_tokenizer = tokenizer
        # Also set is_loaded_in_8bit to disable incorrect DDP
        m.is_loaded_in_8bit = True if not full_finetuning else False

        # Patch generate
        if os.environ.get("UNSLOTH_DISABLE_FAST_GENERATION", "0") == "0":
            if model.generate.__name__ != "unsloth_base_fast_generate":
                model._old_generate = model.generate
                unsloth_base_fast_generate.__doc__ = model._old_generate.__doc__
                model.generate = types.MethodType(unsloth_base_fast_generate, model)
        pass
        model._unsloth_trust_remote_code = trust_remote_code
        # Post patches
        model = FastBaseModel.post_patch_model(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            trust_remote_code  = trust_remote_code,
        )
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        return model, tokenizer
    pass


    @staticmethod
    def get_peft_model(
        model,
        r: int                                          = 16,
        target_modules: Optional[Union[List[str], str]] = None,
        lora_alpha: int                                 = 16,
        lora_dropout: float                             = 0,
        bias: str                                       = "none",
        finetune_vision_layers: bool                    = True,
        finetune_language_layers: bool                  = True,
        finetune_attention_modules: bool                = True,
        finetune_mlp_modules: bool                      = True,
        layers_to_transform: Optional[List[int]]        = None,
        layers_pattern: Optional[str]                   = None,
        use_gradient_checkpointing: bool                = True,
        random_state: int                               = 3407,
        max_seq_length: int                             = 2048, # not used anymore
        use_rslora: bool                                = False,
        modules_to_save: Optional[List[str]]            = None,
        init_lora_weights: bool                         = True,
        loftq_config: Dict[str, Any]                    = {},
        task_type: TaskType                             = TaskType.CAUSAL_LM,
        temporary_location: str                         = "_unsloth_temporary_saved_buffers",
        **kwargs
    ):
        """
        Convert a base model to a PEFT (Parameter-Efficient Fine-Tuning) model.
        
        Args:
            model: The base model to convert
            r (int): Rank for LoRA adaptation
        target_modules (Union[List[str], str], optional): Modules to apply LoRA
        lora_alpha (int): Alpha value for LoRA scaling
        lora_dropout (float): Dropout probability for LoRA layers
        bias (str): Bias configuration for LoRA
        finetune_vision_layers (bool): Whether to fine-tune vision layers
        finetune_language_layers (bool): Whether to fine-tune language layers
        finetune_attention_modules (bool): Whether to fine-tune attention modules
        finetune_mlp_modules (bool): Whether to fine-tune MLP modules
        layers_to_transform (List[int], optional): Specific layers to transform
        layers_pattern (str, optional): Pattern for layer selection
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing
        random_state (int): Random seed for initialization
        max_seq_length (int): Maximum sequence length
        use_rslora (bool): Whether to use rank-stable LoRA
        modules_to_save (List[str], optional): Additional modules to save
        init_lora_weights (bool): Whether to initialize LoRA weights
        loftq_config (Dict[str, Any]): Configuration for LoFTQ quantization
        task_type (TaskType): Type of task for the model
        temporary_location (str): Temporary storage location
        **kwargs: Additional arguments
        
        Returns:
            model: The PEFT-adapted model
        
        This method implements LoRA adaptation with various configuration options,
        including module selection, quantization support, and training
        optimizations. It handles both vision and language model components
        and provides memory-efficient fine-tuning capabilities.
        """
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            return model
        pass
        transformers_set_seed(random_state)

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM):
            raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")

        if target_modules == "all-linear":
            finetune_vision_layers     = True
            finetune_language_layers   = True
            finetune_attention_modules = True
            finetune_mlp_modules       = True
        pass
        if target_modules is None or target_modules == "all-linear":
            target_modules = get_peft_regex(
                model,
                finetune_vision_layers     = finetune_vision_layers,
                finetune_language_layers   = finetune_language_layers,
                finetune_attention_modules = finetune_attention_modules,
                finetune_mlp_modules       = finetune_mlp_modules,
            )
        else:
            assert(type(target_modules) in (list, tuple,))
        pass
        
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        max_seq_length = model.max_seq_length
        lora_config = LoraConfig(
            r               = r,
            lora_alpha      = lora_alpha,
            target_modules  = target_modules,
            lora_dropout    = lora_dropout,
            bias            = bias,
            task_type       = task_type,
            use_rslora      = use_rslora,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )
        model = _get_peft_model(model, lora_config)
        # Enable gradients on modules which are trainable
        requires_grad_for_gradient_checkpointing(model)
        trust_remote_code = getattr(model, "_unsloth_trust_remote_code", False)
        model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing, trust_remote_code = trust_remote_code)
        model.max_seq_length = max_seq_length
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        patch_saving_functions(model, vision = True)

        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        return model
    pass


    @staticmethod
    def post_patch_model(
        model,
        use_gradient_checkpointing: bool = True,
        trust_remote_code: bool          = False,
    ):
        """
        Apply post-processing patches to a model for optimized training and inference.
        
        Args:
            model: The model to patch
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing    trust_remote_code (bool): Whether to trust remote code execution
        
        Returns:
            model: The patched model
        
        This method applies various optimizations including:
        - Gradient checkpointing configuration
        - Training/inference mode setup
        - Tokenizer padding configuration
        - Memory management
        - Hardware-specific optimizations
        - Model-specific patches
        
        It ensures the model is properly configured for both training and
        inference scenarios with optimized performance characteristics.
        """
        full_finetuning = os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"

        float32_mixed_precision = True
        if _get_dtype(model.config.torch_dtype) == torch.bfloat16 and full_finetuning:
            # Use bfloat16 precision for full finetuning
            float32_mixed_precision = False

        model = prepare_model_for_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant              = True,
            full_finetuning            = full_finetuning,
            train_layernorms           = full_finetuning,
            train_embedding            = full_finetuning,
            train_lm_head              = full_finetuning,
            float32_mixed_precision    = float32_mixed_precision,
        )

        from transformers.trainer import Trainer 
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop" and trust_remote_code == False:
            raise RuntimeError('Unsloth: Unsuccessfully patched inner_training_loop')
        pass
        patch_saving_functions(model, vision = True)

        # Patch tokenizer to pad to the right
        m = model
        while hasattr(m, "model"):
            if hasattr(m, "_saved_temp_tokenizer"):
                if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                    m._saved_temp_tokenizer.tokenizer.padding_side = "right"
            pass
            # Also set is_loaded_in_8bit to disable incorrect DDP
            m.is_loaded_in_8bit = True if not full_finetuning else False
            m = m.model
        pass
        if hasattr(m, "_saved_temp_tokenizer"):
            if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                m._saved_temp_tokenizer.tokenizer.padding_side = "right"
        pass
        # Also set is_loaded_in_8bit to disable incorrect DDP
        m.is_loaded_in_8bit = True if not full_finetuning else False

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        return model
    pass


    @staticmethod
    def for_inference(model):
        """
        Configure a model for inference mode with optimized settings.
        
        Args:
            model: The model to configure
        
        Returns:
            model: The model configured for inference
        
        This method disables training-specific settings and enables
        optimizations for inference including:
        - Disabling gradient computation
        - Setting evaluation mode
        - Configuring padding side
        - Enabling generation flags
        - Disabling training-specific components
        
        It ensures the model is properly configured for efficient
        inference execution.
        """
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_inference!")

        def _for_inference(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = False
            if hasattr(m, "training"): m.training = False
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "left"
            # Set a flag for generation!
            m._flag_for_generation = True
        pass
        m = model
        while hasattr(m, "model"):
            _for_inference(m)
            m = m.model
        _for_inference(m)

        # Also disable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        return model
    pass


    @staticmethod
    def for_training(model, use_gradient_checkpointing: bool = True):
        """
        Configure a model for training mode with optimized settings.
        
        Args:
            model: The model to configure
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing
        
        Returns:
            model: The model configured for training
        
        This method enables training-specific settings and optimizations
        including:
        - Enabling gradient computation
        - Setting training mode
        - Configuring padding side
        - Disabling generation flags
        - Enabling training-specific components
        - Gradient checkpointing configuration
        
        It ensures the model is properly configured for efficient
        training execution with memory optimizations.
        """
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_training!")

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora
        pass

        def _for_training(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(m, "training"): m.training = True
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "right"
            # Set a flag for generation!
            if hasattr(m, "_flag_for_generation"): del m._flag_for_generation
        pass
        m = model
        while hasattr(m, "model"):
            _for_training(m)
            m = m.model
        _for_training(m)

        # Also re-enable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        return model
    pass
pass