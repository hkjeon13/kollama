from typing import Literal

from transformers import PreTrainedModel

from .params import LoraParams

try:
    from peft import PeftModel, LoraConfig, TaskType, get_peft_model
except ImportError:
    raise ImportError("Please install peft library to apply PEFT LoRA (e.g. $pip install peft)")


def get_lora_model(
        model: PreTrainedModel,
        lora_config: LoraParams,
        model_type: Literal["causal", "seq2seq"] = "causal",
        print_trainable_parameters: bool = True,
) -> PeftModel:
    """
    LoRA를 적용한 모델을 반환
    :param print_trainable_parameters:
    :param model:
    :param lora_config:
    :param model_type:
    :return:

    >>> from transformers import AutoModelForCausalLM
    >>> model = AutoModelForCausalLM.from_pretrained("psyche/kogpt")
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class LoraParams:
    ...     apply_lora: bool = True
    ...     lora_r: float = 64
    ...     lora_alpha: float = 16
    ...     lora_dropout: float = 0.1
    >>> lora_config = LoraParams()
    >>> model = get_lora_model(model, lora_config, model_type="causal", print_trainable_parameters=False)
    >>> model.__class__.__name__
    'PeftModelForCausalLM'
    """

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM if model_type == "causal" else TaskType.SEQ_2_SEQ_LM,
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
    )

    model = get_peft_model(model, lora_config)

    if print_trainable_parameters:
        model.print_trainable_parameters()

    return model
