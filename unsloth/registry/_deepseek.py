from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_DEEPSEEK_V3_REGISTERED = False
_IS_DEEPSEEK_V3_0324_REGISTERED = False
_IS_DEEPSEEK_R1_REGISTERED = False
_IS_DEEPSEEK_R1_ZERO_REGISTERED = False
_IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED = False
_IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED = False

class DeepseekV3ModelInfo(ModelInfo):
    """
    Model information class for DeepSeek V3 models.
    
    This class provides methods for constructing model names based on base name, version, size, quantization type, and instruction tags.
    
    Args:
        base_name (`str`):
            The base name of the model.
        version (`str`):
            The version of the model.
        size (`str`):
            The size of the model.
        quant_type (`QuantType`):
            The quantization type of the model.
        instruct_tag (`str`):
            The instruction tag associated with the model.
    """
    @classmethod
    def construct_model_name(cls, base_name: str, version: str, size: str, quant_type: QuantType, instruct_tag: str) -> str:
        """
        Constructs the model name for DeepSeek V3 models.
        
        Args:
            base_name (`str`):
                The base name of the model.
            version (`str`):
                The version of the model.
            size (`str`):
                The size of the model.
            quant_type (`QuantType`):
                The quantization type of the model.
            instruct_tag (`str`):
                The instruction tag associated with the model.
            key (`str`):
                The key used to construct the model name.
        
        Returns:
            `str`: The constructed model name.
        """
        key = f"{base_name}-V{version}"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)

class DeepseekR1ModelInfo(ModelInfo):
    """
    Model information class for DeepSeek R1 models.
    
    This class provides methods for constructing model names based on base name, version, size, quantization type, and instruction tags.
    
    Args:
        base_name (`str`):
            The base name of the model.
        version (`str`):
            The version of the model.
        size (`str`):
            The size of the model.
        quant_type (`QuantType`):
            The quantization type of the model.
        instruct_tag (`str`):
            The instruction tag associated with the model.
    """
    @classmethod
    def construct_model_name(cls, base_name: str, version: str, size: str, quant_type: QuantType, instruct_tag: str) -> str:
        """
        Constructs the model name for DeepSeek R1 models.
        
        Args:
            base_name (`str`):
                The base name of the model.
            version (`str`):
                The version of the model.
            size (`str`):
                The size of the model.
            quant_type (`QuantType`):
                The quantization type of the model.
            instruct_tag (`str`):
                The instruction tag associated with the model.
            key (`str`):
                The key used to construct the model name.
        
        Returns:
            `str`: The constructed model name.
        """
        key = f"{base_name}-{version}" if version else base_name
        if size:
            key = f"{key}-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)
    
# Deepseek V3 Model Meta
DeepseekV3Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    instruct_tags=[None],
    model_version="3",
    model_sizes=[""],
    model_info_cls=DeepseekV3ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BF16],
)

DeepseekV3_0324Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    instruct_tags=[None],
    model_version="3-0324",
    model_sizes=[""],
    model_info_cls=DeepseekV3ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.GGUF],
)

DeepseekR1Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1",
    instruct_tags=[None],
    model_version="",
    model_sizes=[""],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BF16, QuantType.GGUF],
)

DeepseekR1ZeroMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1",
    instruct_tags=[None],
    model_version="Zero",
    model_sizes=[""],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.GGUF],
)

DeepseekR1DistillLlamaMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1-Distill",
    instruct_tags=[None],
    model_version="Llama",
    model_sizes=["8", "70"],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types={"8": [QuantType.UNSLOTH, QuantType.GGUF], "70": [QuantType.GGUF]},
)

# Deepseek R1 Distill Qwen Model Meta
DeepseekR1DistillQwenMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1-Distill",
    instruct_tags=[None],
    model_version="Qwen",
    model_sizes=["1.5", "7", "14", "32"],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types={
        "1.5": [QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF],
        "7": [QuantType.UNSLOTH, QuantType.BNB],
        "14": [QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF],
        "32": [QuantType.GGUF, QuantType.BNB],
    },
)
        
def register_deepseek_v3_models(include_original_model: bool = False) -> None:
    """
    Registers DeepSeek V3 models with the model registry.
    
    Args:
        include_original_model (`bool`, optional):
            Whether to include the original model in the registration. Defaults to False.
    
    Returns:
        None
    """
    global _IS_DEEPSEEK_V3_REGISTERED
    if _IS_DEEPSEEK_V3_REGISTERED:
        return
    _register_models(DeepseekV3Meta, include_original_model=include_original_model)
    _IS_DEEPSEEK_V3_REGISTERED = True

def register_deepseek_v3_0324_models(include_original_model: bool = False) -> None:
    """
    Registers DeepSeek V3 0324 models with the model registry.
    
    Args:
        include_original_model (`bool`, optional):
            Whether to include the original model in the registration. Defaults to False.
    
    Returns:
        None
    """
    global _IS_DEEPSEEK_V3_0324_REGISTERED
    if _IS_DEEPSEEK_V3_0324_REGISTERED:
        return
    _register_models(DeepseekV3_0324Meta, include_original_model=include_original_model)
    _IS_DEEPSEEK_V3_0324_REGISTERED = True

def register_deepseek_r1_models(include_original_model: bool = False) -> None:
    """
    Registers DeepSeek R1 models with the model registry.
    
    Args:
        include_original_model (`bool`, optional):
            Whether to include the original model in the registration. Defaults to False.
    
    Returns:
        None
    """
    global _IS_DEEPSEEK_R1_REGISTERED
    if _IS_DEEPSEEK_R1_REGISTERED:
        return
    _register_models(DeepseekR1Meta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_REGISTERED = True

def register_deepseek_r1_zero_models(include_original_model: bool = False) -> None:
    """
    Registers DeepSeek R1 Zero models with the model registry.
    
    Args:
        include_original_model (`bool`, optional):
            Whether to include the original model in the registration. Defaults to False.
    
    Returns:
        None
    """
    global _IS_DEEPSEEK_R1_ZERO_REGISTERED
    if _IS_DEEPSEEK_R1_ZERO_REGISTERED:
        return
    _register_models(DeepseekR1ZeroMeta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_ZERO_REGISTERED = True

def register_deepseek_r1_distill_llama_models(include_original_model: bool = False) -> None:
    """
    Registers DeepSeek R1 Distill Llama models with the model registry.
    
    Args:
        include_original_model (`bool`, optional):
            Whether to include the original model in the registration. Defaults to False.
    
    Returns:
        None
    """
    global _IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED
    if _IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED:
        return
    _register_models(DeepseekR1DistillLlamaMeta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED = True

def register_deepseek_r1_distill_qwen_models(include_original_model: bool = False) -> None:
    """
    Registers DeepSeek R1 Distill Qwen models with the model registry.
    
    Args:
        include_original_model (`bool`, optional):
            Whether to include the original model in the registration. Defaults to False.
    
    Returns:
        None
    """
    global _IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED
    if _IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED:
        return
    _register_models(DeepseekR1DistillQwenMeta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED = True

def register_deepseek_models(include_original_model: bool = False) -> None:
    """
    Registers all DeepSeek models with the model registry.
    
    Args:
        include_original_model (`bool`, optional):
            Whether to include the original models in the registration. Defaults to False.
    
    Returns:
        None
    """
    register_deepseek_v3_models(include_original_model=include_original_model)
    register_deepseek_v3_0324_models(include_original_model=include_original_model)
    register_deepseek_r1_models(include_original_model=include_original_model)
    register_deepseek_r1_zero_models(include_original_model=include_original_model)
    register_deepseek_r1_distill_llama_models(include_original_model=include_original_model)
    register_deepseek_r1_distill_qwen_models(include_original_model=include_original_model)

def _list_deepseek_r1_distill_models() -> list[str]:
    """
    Lists all DeepSeek R1 Distill models available on the Hugging Face Hub.
    
    Returns:
        `list[str]`: A list of model versions for DeepSeek R1 Distill models.
    """
    from unsloth.utils.hf_hub import ModelInfo as HfModelInfo
    from unsloth.utils.hf_hub import list_models
    models: list[HfModelInfo] = list_models(author="unsloth", search="Distill", limit=1000)
    distill_models = []
    for model in models:
        model_id = model.id
        model_name = model_id.split("/")[-1]
        # parse out only the version
        version = model_name.removeprefix("DeepSeek-R1-Distill-")
        distill_models.append(version)

    return distill_models


register_deepseek_models(include_original_model=True)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    MODEL_REGISTRY.clear()
    
    register_deepseek_models(include_original_model=True)
    
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
    # distill_models = _list_deepseek_r1_distill_models()
    # for model in sorted(distill_models):
    #     if "qwen" in model.lower():
    #         print(model)