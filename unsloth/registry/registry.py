from typing import Optional
import warnings
from dataclasses import dataclass, field
from enum import Enum


class QuantType(Enum):
    """
    Enumeration of supported quantization types for models.

    Values:
            BNB: 4-bit quantization using bitsandbytes.
            UNSLOTH: Dynamic 4-bit quantization using Unsloth.
            GGUF: Quantization using GGUF format.
            NONE: No quantization applied.
            BF16: BFloat16 precision (used for specific models like Deepseek V3).
    """

    BNB: str = "bnb"
    UNSLOTH: str = "unsloth"  # dynamic 4-bit quantization
    GGUF: str = "GGUF"
    NONE: str = "none"
    BF16: str = "bf16"  # only for Deepseek V3


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
    Data class containing metadata about a model.

    Args:
            org (`str`): Organization or user name on Hugging Face Hub.
            base_name (`str`): Base name of the model.
            version (`str`): Version of the model.
            size (`int`): Size of the model in MB or GB.
            name (`str`, optional): Full model name. If not provided, constructed from other fields.
            is_multimodal (`bool`, optional): Whether the model supports multimodal inputs.
            instruct_tag (`str`, optional): Tag indicating instruction-tuned version.
            quant_type (`QuantType`, optional): Quantization type used for the model.
            description (`str`, optional): Description of the model.
    """

    org: str
    base_name: str
    version: str
    size: int
    name: str = (
        None  # full model name, constructed from base_name, version, and size unless provided
    )
    is_multimodal: bool = False
    instruct_tag: str = None
    quant_type: QuantType = None
    description: str = None

    def __post_init__(self):
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
        Adds instruction-tuning tag to a model name key if provided.

        Args:
                key (`str`): Base key to append to.
                instruct_tag (`str`, optional): Instruction-tuning tag to append.

        Returns:
                `str`: Key with instruction tag appended if present.
        """
        if instruct_tag:
            key = "-".join([key, instruct_tag])
        return key

    @staticmethod
    def append_quant_type(key: str, quant_type: QuantType = None) -> str:
        """
        Adds quantization type tag to a model name key if quantization is applied.

        Args:
                key (`str`): Base key to append to.
                quant_type (`QuantType`, optional): Quantization type to append as tag.

        Returns:
                `str`: Key with quantization tag appended if applicable.
        """
        if quant_type != QuantType.NONE:
            key = "-".join([key, QUANT_TAG_MAP[quant_type]])
        return key

    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType,
        instruct_tag: Optional[str],
        key: str = "",
    ) -> str:
        """
        Constructs a complete model name from components.

        Args:
                base_name (`str`): Base name of the model.
                version (`str`): Version of the model.
                size (`str`): Size indicator of the model.
                quant_type (`QuantType`, optional): Quantization type to include in name.
                instruct_tag (`str`, optional): Instruction-tuning tag to include.
                key (`str`, optional): Optional base key to build upon.

        Returns:
                `str`: Constructed model name combining all components.
        """
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

    @property
    def model_path(
        self,
    ) -> str:
        """
        Constructs the full model path for Hugging Face Hub.

        Returns:
                `str`: Full model path in format '{org}/{name}'.
        """
        return f"{self.org}/{self.name}"


@dataclass
class ModelMeta:
    """
    Data class containing metadata for model registration.

    Args:
            org (`str`): Organization name on Hugging Face Hub.
            base_name (`str`): Base name for all models in this group.
            model_version (`str`): Version for all models in this group.
            model_info_cls (`type[ModelInfo]`): Class to use for creating ModelInfo instances.
            model_sizes (`list[str]`): List of model sizes to register.
            instruct_tags (`list[str]`): List of instruction-tuning tags to register.
            quant_types (`list[QuantType]` or `dict[str, list[QuantType]]`): Quantization types to register,
                    can be different per model size when using dictionary format.
            is_multimodal (`bool`, optional): Whether models in this group are multimodal.
    """

    org: str
    base_name: str
    model_version: str
    model_info_cls: type[ModelInfo]
    model_sizes: list[str] = field(default_factory=list)
    instruct_tags: list[str] = field(default_factory=list)
    quant_types: list[QuantType] | dict[str, list[QuantType]] = field(
        default_factory=list
    )
    is_multimodal: bool = False


MODEL_REGISTRY: dict[str, ModelInfo] = {}


def register_model(
    model_info_cls: ModelInfo,
    org: str,
    base_name: str,
    version: str,
    size: int,
    instruct_tag: str = None,
    quant_type: QuantType = None,
    is_multimodal: bool = False,
    name: str = None,
) -> None:
    """
    Registers a model in the global MODEL_REGISTRY.

    Args:
            model_info_cls (`ModelInfo`): Class to use for creating the model info.
            org (`str`): Organization name on Hugging Face Hub.
            base_name (`str`): Base name of the model.
            version (`str`): Version of the model.
            size (`int`): Size of the model in MB or GB.
            instruct_tag (`str`, optional): Instruction-tuning tag for the model.
            quant_type (`QuantType`, optional): Quantization type used for the model.
            is_multimodal (`bool`, optional): Whether the model supports multimodal inputs.
            name (`str`, optional): Full model name. If not provided, constructed from other fields.
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
        raise ValueError(
            f"Model {key} already registered, current keys: {MODEL_REGISTRY.keys()}"
        )

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


def _check_model_info(
    model_id: str, properties: list[str] = ["lastModified"]
) -> Optional[ModelInfo]:
    """
    Checks if a model exists on Hugging Face Hub and retrieves its info.

    Args:
            model_id (`str`): Identifier of the model in format '{org}/{name}'.
            properties (`list[str]`, optional): List of properties to retrieve from the model info.

    Returns:
            `Optional[ModelInfo]`: Hugging Face ModelInfo object if model exists, None otherwise.
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


def _register_models(
    model_meta: ModelMeta, include_original_model: bool = False
) -> None:
    """
    Registers multiple models based on ModelMeta configuration.

    Args:
            model_meta (`ModelMeta`): Configuration object containing registration parameters.
            include_original_model (`bool`, optional): Whether to include the original unquantized model from the releasing organization.
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
                _org = "unsloth"  # unsloth models -- these are all quantized versions of the original model
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
