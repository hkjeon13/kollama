from typing import Any, Dict, Optional,Literal
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)


@dataclass
class ModelParams:
    model_name_or_path: str = field(
        metadata={"help": "The model name or path"}
    )

    revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use"}
    )

    model_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "The model auth token"}
    )

    model_type: Literal["causal", "seq2seq"] = field(
        default="causal",
        metadata={"help": "The model type"}
    )

    add_pad_token: bool = field(
        default=False,
        metadata={"help": "Whether to add pad token"}
    )


@dataclass
class LoraParams:
    apply_lora: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA"}
    )

    lora_r: float = field(
        default=64,
        metadata={"help": "The r value of LoRA"}
    )

    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha value of LoRA"}
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout value of LoRA"}
    )

@dataclass
class BnBParams:
    apply_4bit_training: bool = field(
        default=False,
        metadata={"help": "Whether to apply 4-bit training with bitsandbytes"}
    )

    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use double quantization for BnB 4bit"}
    )

    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "The quantization type for BnB 4bit"}
    )

    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "The compute dtype for BnB 4bit"}
    )

@dataclass
class DataParams:
    data_name_or_path: str = field(
        metadata={"help": "The data name or path"}
    )

    data_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "The data auth token"}
    )

    max_seq_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization"}
    )

    train_split_name: str = field(
        default="train",
        metadata={"help": "The name of train split"}
    )

    eval_split_name: str = field(
        default="validation",
        metadata={"help": "The name of eval split"}
    )


def get_lora_model(
        model: PreTrainedModel,
        lora_config: LoraParams,
        model_type: Literal["causal", "seq2seq"] = "causal",
        print_trainable_parameters: bool = True,
):
    """
    Get LoRA model
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
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError("Please install peft library to apply PEFT LoRA (e.g. $pip install peft)")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM if model_type == "causal" else TaskType.SEQ2SEQ,
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
    )

    model = get_peft_model(model, lora_config)
    if print_trainable_parameters:
        model.print_trainable_parameters()
    return model


def get_bnb_config(bnb_config: BnBParams) -> Dict[str, Any]:
    """
    Get additional config for bitsandbytes
    :param bnb_config:
    :return:
        - additional_config: Dict[str, Any]
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class BnBParams:
    ...     apply_4bit_training: bool = True
    ...     bnb_4bit_use_double_quant: bool = False
    ...     bnb_4bit_quant_type: str = "nf4"
    ...     bnb_4bit_compute_dtype: str = "float16"
    >>> bnb_config = BnBParams()
    >>> get_bnb_config(bnb_config)["quantization_config"].__class__.__name__
    'BitsAndBytesConfig'
    """
    additional_config = {}
    if bnb_config.apply_4bit_training:
        import torch
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, bnb_config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        additional_config["quantization_config"] = bnb_config

    return additional_config


def get_data_collator(model, tokenizer, model_args):
    """
    Get data collator
    :param model:
    :param tokenizer:
    :param model_args:
    :return:

    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> tokenizer = AutoTokenizer.from_pretrained("psyche/kogpt")
    >>> model = AutoModelForCausalLM.from_pretrained("psyche/kogpt")
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class ModelParams:
    ...     model_type: str = "causal"
    >>> model_args = ModelParams()
    >>> get_data_collator(model, tokenizer, model_args).__class__.__name__
    'DataCollatorForLanguageModeling'
    """
    from inspect import signature

    collator_module = DataCollatorForLanguageModeling \
        if model_args.model_type == "causal" else DataCollatorForSeq2Seq
    candidates = {
        "tokenizer": tokenizer, "model": model, "mlm": False,
        "pad_to_multiple_of": None
    }
    para_names = signature(collator_module.__init__).parameters.keys()
    collator_args = {
        key: candidates.get(key) for key in para_names if key in candidates
    }

    return collator_module(**collator_args)


def get_callbacks(model_args):
    trainer_callbacks = None
    if model_args.use_slack_notifier:
        try:
            from utils.callbacks import SlackOnLogCallback
        except ImportError:
            raise ImportError("Please install slacker to use slack notifier callback (e.g. $pip install slacker)")

        trainer_callbacks = [
            SlackOnLogCallback(
                slack_token=model_args.slack_token,
                message_channel=model_args.slack_channel,
                message_prefix=model_args.slack_message_prefix,
            )
        ]

    return trainer_callbacks


def main():
    parser = HfArgumentParser((ModelParams, DataParams, TrainingArguments, LoraParams, BnBParams))
    model_args, data_args, training_args, lora_config, bnb_config = parser.parse_args_into_dataclasses()

    additional_config = get_bnb_config(bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.revision,
        use_auth_token=model_args.model_auth_token,
        **additional_config
    )


    model_class = AutoModelForCausalLM \
        if model_args.model_type == "causal" else AutoModelForSeq2SeqLM

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.revision,
        use_auth_token=model_args.model_auth_token
    )

    if lora_config.apply_lora:
        model = get_lora_model(
            model, model_args, model_type=model_args.model_type
        )

    if model_args.add_pad_token:
        from utils.tokenizer_utils import get_special_tokens
        _sample_sp_token = list(tokenizer.special_tokens_map.values())[0]
        tokenizer.add_special_tokens({"pad_token": get_special_tokens(_sample_sp_token, "pad")})
        model = model.resize_token_embeddings(len(tokenizer))

    data_collator = get_data_collator(model, tokenizer, model_args)

    if bnb_config.do_4bit_training:
        training_args.optim = "paged_adamw_32bit"

    trainer_callbacks = get_callbacks(model_args)
    # TODO: add data download
    
    dataset = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[data_args.train_split_name],
        eval_dataset=dataset[data_args.eval_split_name],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=trainer_callbacks,
    )

    if training_args.do_train:
        trainer.train()

    elif training_args.do_eval:
        result = trainer.evaluate()
        print("***** Eval results *****")
        for key, value in result.items():
            print(f"  {key} = {value:.3f}")












if __name__ == "__main__":
    main()