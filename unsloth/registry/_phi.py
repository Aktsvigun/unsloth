from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_PHI_4_REGISTERED = False
_IS_PHI_4_INSTRUCT_REGISTERED = False


class PhiModelInfo(ModelInfo):
    """
    Class representing information about Phi models. This class is used to construct model names based on various parameters such as base name, version, size, quantization type, and instruction tag.
    """

    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType,
        instruct_tag: str,
    ) -> str:
        """
        Constructs a model name based on the provided parameters.

        Args:
                base_name (`str`): The base name of the model.
                version (`str`): The version of the model.
                size (`str`): The size of the model.
                quant_type (`QuantType`): The quantization type of the model.
                instruct_tag (`str`): The instruction tag of the model.
                key (`str`): A key used in the model name construction.

        Returns:
                `str`: The constructed model name.
        """
        key = f"{base_name}-{version}"
        return super().construct_model_name(
            base_name, version, size, quant_type, instruct_tag, key
        )


# Phi Model Meta
PhiMeta4 = ModelMeta(
    org="microsoft",
    base_name="phi",
    instruct_tags=[None],
    model_version="4",
    model_sizes=["1"],  # Assuming only one size
    model_info_cls=PhiModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Phi Instruct Model Meta
PhiInstructMeta4 = ModelMeta(
    org="microsoft",
    base_name="phi",
    instruct_tags=["mini-instruct"],
    model_version="4",
    model_sizes=["1"],  # Assuming only one size
    model_info_cls=PhiModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH, QuantType.GGUF],
)


def register_phi_4_models(include_original_model: bool = False) -> None:
    """
    Registers Phi 4 models in the model registry. This function ensures that the models are registered only once.

    Args:
            include_original_model (`bool`, optional): Whether to include the original model in the registration. Defaults to False.
    """
    global _IS_PHI_4_REGISTERED
    if _IS_PHI_4_REGISTERED:
        return
    _register_models(PhiMeta4, include_original_model=include_original_model)
    _IS_PHI_4_REGISTERED = True


def register_phi_4_instruct_models(include_original_model: bool = False) -> None:
    """
    Registers Phi 4 instruct models in the model registry. This function ensures that the models are registered only once.

    Args:
            include_original_model (`bool`, optional): Whether to include the original model in the registration. Defaults to False.
    """
    global _IS_PHI_4_INSTRUCT_REGISTERED
    if _IS_PHI_4_INSTRUCT_REGISTERED:
        return
    _register_models(PhiInstructMeta4, include_original_model=include_original_model)
    _IS_PHI_4_INSTRUCT_REGISTERED = True


def register_phi_models(include_original_model: bool = False) -> None:
    """
    Registers all Phi models (both standard and instruct versions) in the model registry.

    Args:
            include_original_model (`bool`, optional): Whether to include the original model in the registration. Defaults to False.
    """
    register_phi_4_models(include_original_model=include_original_model)
    register_phi_4_instruct_models(include_original_model=include_original_model)


if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info

    MODEL_REGISTRY.clear()

    register_phi_models(include_original_model=True)

    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
