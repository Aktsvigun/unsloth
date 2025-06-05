from typing import Optional
import warnings
from dataclasses import dataclass, field
from enum import Enum


class QuantType(Enum):
    """
    An enumeration class representing different types of quantization methods.
    
    Attributes:
        BNB (`str`):
            BitsAndBytes quantization method.
        UNSLOTH (`str`):
            Dynamic 4-bit quantization method.
        GGUF (`str`):
            GGUF quantization method.
        NONE (`str`):
            No quantization applied.
        BF16 (`str`):
            Brain Floating Point 16-bit quantization method, used only for Deepseek V3.
    """
    BNB: str  = "bnb"

    UNSLOTH: str = "unsloth" # dynamic 4-bit quantization
    GGUF: str = "GGUF"

    NONE: str = "none"

    BF16: str = "bf16" # only for Deepseek V3

# Tags for Hugging Face model paths
BNB_QUANTIZED_TAG = "bnb-4bit"
UNSLOTH_DYNAMIC_QUANT_TAG = "unsloth" + "-" + BNB_QUANTIZED_TAG
GGUF_TAG = "GGUF"
BF16_TAG = "bf16"

QUANT_TAG_MAP = {
    QuantType.BNB: BNB_QUANTIZED_TAG,
    QuantType.UNSLOTH: UNSLOTH_DYNAMIC_QUANT_TAG,
    QuantType.GGUF: GGUF_TAG,
    QuantType.NONE: None,
    QuantType.BF16: BF16_TAG,
} 

# NOTE: models registered with org="unsloth" and QUANT_TYPE.NONE are aliases of QUANT_TYPE.UNSLOTH
@dataclass
class ModelInfo:
    """
    A data class that holds information about a model.
    
    Attributes:
        or` (`str`):
            The organization that owns the model.
        base_name (`str`):
            The base name of the model.
        version (`str`):
            The version of the model.    size (`int`):
            The size of the model.    name (`str`, optional):
            The full model name, constructed from base_name, version, and size unless provided.    is_multimodal (`bool`):
            Whether the model is multimodal.    instruct_tag (`str`, optional):
            Instruction tag for the model.    quant_type (`QuantType`, optional):
            The quantization type of the model.    description (`str`, optional):
            Description of the model.
    
    Methods:
        __post_init__():
            Constructs the model name if it is not provided.    append_instruct_tag(key: str, instruct_tag: str):
            Appends the instruction tag to the model name.    append_quant_type(key: str, quant_type: QuantType):
            Appends the quantization type to the model name.    construct_model_name(base_name: str, version: str, size: int, quant_type: QuantType, instruct_tag: str, key: str):
            Constructs the model name from the base name, version, size, quantization type, and instruction tag.    model_path():
            Returns the model path in the format 'org/name'.
    """
    org: str
    base_name: str
    version: str
    size: int
    name: str = None  # full model name, constructed from base_name, version, and size unless provided
    is_multimodal: bool   = False

    instruct_tag: str     = None

    quant_type: QuantType = None

    description: str      = None


    def __post_init__(self):
        """
        Constructs the model name if it is not provided during initialization.
        """
        self.name = self.name or self.construct_model_name(
            self.base_name,
            self.version,
            self.size,
            self.quant_type,
            self.instruct_tag,
        )

    @staticmethod
    def append_instruct_tag(key: str, instruct_tag: str = None) -> str:
        """
        Appends the instruction tag to the model name.
        
        Args:
            key (`str`):
                The base model name.    instruct_tag (`str`, optional):
                The instruction tag to append.
        
        Returns:
            `str`: The model name with the instruction tag appended.
        """
        if instruct_tag:
            key = "-".join([key, instruct_tag])
        return key

    @staticmethod
    def append_quant_type(
        key: str, quant_type: QuantType = None
    ) -> str:
        """
        Appends the quantization type to the model name.
        
        Args:
            key (`str`):
                The base model name.    quant_type (`QuantType`, optional):
                The quantization type to append.
        
        Returns:
            `str`: The model name with the quantization type appended.
        """
        if quant_type != QuantType.NONE:
            key = "-".join([key, QUANT_TAG_MAP[quant_type]])
        return key

    @classmethod
    def construct_model_name(cls, base_name: str, version: str, size: int, quant_type: QuantType, instruct_tag: str, key: str="") -> str:
        """
        Constructs the model name from the base name, version, size, quantization type, and instruction tag.
        
        Args:
            base_name (`str`):
                The base name of the model.    version (`str`):
                The version of the model.    size (`int`):
                The size of the model.    quant_type (`QuantType`):
                The quantization type of the model.    instruct_tag (`str`):
                The instruction tag of the model.    key (`str`, optional):
                The base model name to build upon.
        
        Returns:
            `str`: The full model name.
        """
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

    @property
    def model_path(
        self,
    ) -> str:
        """
        Returns the model path in the format 'org/name'.
        
        Returns:
            `str`: The model path.
        """
        return f"{self.org}/{self.name}"


@dataclass
class ModelMeta:
    """
    A data class that holds metadata for a model.
    
    Attributes:
        or` (`str`):
            The organization that owns the model.    base_name (`str`):
            The base name of the model.    model_version (`str`):
            The version of the model.    model_info_cls (`type[ModelInfo]`):
            The class used to create model information objects.    model_sizes (`list[str]`):
            The sizes of the model.    instruct_tags (`list[str]`):
            The instruction tags for the model.    quant_types (`list[QuantType]` or `dict[str, list[QuantType]]`):
            The quantization types for the model.    is_multimodal (`bool`):
            Whether the model is multimodal.
    """
    org: str
    base_name: str
    model_version: str
    model_info_cls: type[ModelInfo]
    model_sizes: list[str]   = field(default_factory=list)

    instruct_tags: list[str] = field(default_factory=list)

    quant_types: list[QuantType] | dict[str, list[QuantType]] = field(default_factory=list)
    is_multimodal: bool      = False



MODEL_REGISTRY: dict[str, ModelInfo] = {}


def register_model(
    model_info_cls: ModelInfo,
    org: str,
    base_name: str,
    version: str,
    size: int,
    instruct_tag: str     = None,
    quant_type: QuantType = None,
    is_multimodal: bool   = False,
    name: str             = None,
) -> None:
    """
    Registers a model in the MODEL_REGISTRY.
    
    Args:
        model_info_cls (`ModelInfo`):
            The model information class.    or` (`str`):
            The organization that owns the model.    base_name (`str`):
            The base name of the model.    version (`str`):
            The version of the model.    size (`int`):
            The size of the model.    instruct_tag (`str`, optional):
            The instruction tag of the model.    quant_type (`QuantType`, optional):
            The quantization type of the model.    is_multimodal (`bool`, optional):
            Whether the model is multimodal.    name (`str`, optional):
            The full model name.
    
    Raises:
        ValueError: If the model is already registered.
    """
    name = name or model_info_cls.construct_model_name(
        base_name=base_name,
        version=version,
        size=size,
        quant_type=quant_type,
        instruct_tag=instruct_tag,
    )
    key = f"{org}/{name}"

    if key in MODEL_REGISTRY:
        raise ValueError(f"Model {key} already registered, current keys: {MODEL_REGISTRY.keys()}")

    MODEL_REGISTRY[key] = model_info_cls(
        org=org,
        base_name=base_name,
        version=version,
        size=size,
        is_multimodal=is_multimodal,
        instruct_tag=instruct_tag,
        quant_type=quant_type,
        name=name,
    )


def _check_model_info(model_id: str, properties: list[str] = ["lastModified"]) -> Optional[HfModelInfo]:
    """
    Checks if a model exists on Hugging Face.
    
    Args:
        model_id (`str`):
            The ID of the model to check.    properties (`list[str]`, optional):
            The properties to expand for the model.
    
    Returns:
        `Optional[HfModelInfo]`: The model information if it exists, otherwise None.
    """
    from huggingface_hub import HfApi
    from huggingface_hub import ModelInfo as HfModelInfo
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()

    try:
        model_info: HfModelInfo = api.model_info(model_id, expand=properties)
    except Exception as e:
        if isinstance(e, RepositoryNotFoundError):
            warnings.warn(f"{model_id} not found on Hugging Face")
            model_info = None
        else:
            raise e
    return model_info


def _register_models(model_meta: ModelMeta, include_original_model: bool = False) -> None:
    """
    Registers multiple models in the MODEL_REGISTRY.
    
    Args:
        model_meta (`ModelMeta`):
            The metadata for the models to register.    include_original_model (`bool`, optional):
            Whether to include the original model from the releasing organization.
    """
    org = model_meta.org
    base_name = model_meta.base_name
    instruct_tags = model_meta.instruct_tags
    model_version = model_meta.model_version
    model_sizes = model_meta.model_sizes
    is_multimodal = model_meta.is_multimodal
    quant_types = model_meta.quant_types
    model_info_cls = model_meta.model_info_cls

    for size in model_sizes:
        for instruct_tag in instruct_tags:
            # Handle quant types per model size
            if isinstance(quant_types, dict):
                _quant_types = quant_types[size]
            else:
                _quant_types = quant_types
            for quant_type in _quant_types:
                # NOTE: models registered with org="unsloth" and QUANT_TYPE.NONE are aliases of QUANT_TYPE.UNSLOTH
                _org = "unsloth" # unsloth models -- these are all quantized versions of the original model
                register_model(
                    model_info_cls=model_info_cls,
                    org=_org,
                    base_name=base_name,
                    version=model_version,
                    size=size,
                    instruct_tag=instruct_tag,
                    quant_type=quant_type,
                    is_multimodal=is_multimodal,
                )
            # include original model from releasing organization
            if include_original_model:
                register_model(
                    model_info_cls=model_info_cls,
                    org=org,
                    base_name=base_name,
                    version=model_version,
                    size=size,
                    instruct_tag=instruct_tag,
                    quant_type=QuantType.NONE,
                    is_multimodal=is_multimodal,
                )
