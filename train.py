from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal, Union

import datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

from dataloader import load
from utils import get_tokenized_dataset, GenerationParams, get_callbacks, get_data_collator, get_special_tokens


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

    lora_r: int = field(
        default=64,
        metadata={"help": "The r value of LoRA"}
    )

    lora_alpha: int = field(
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

    max_input_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization"}
    )

    max_output_length: int = field(
        default=1024,
        metadata={"help": "The maximum total output sequence length after tokenization"}
    )

    train_split_name: str = field(
        default="train",
        metadata={"help": "The name of train split"}
    )

    eval_split_name: str = field(
        default="validation",
        metadata={"help": "The name of eval split"}
    )

    is_supervised_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to use supervised dataset"}
    )

    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming dataset"}
    )

    train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The number of train samples to use"}
    )

    eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The number of eval samples to use"}
    )

    prefix: str = field(
        default="",
        metadata={"help": "The prefix to add to the input"}
    )

    suffix: str = field(
        default="",
        metadata={"help": "The suffix to add to the input"}
    )

    input_column_name: str = field(
        default="input",
        metadata={"help": "The column to choose from the dataset"}
    )

    output_column_name: str = field(
        default="output",
        metadata={"help": "The column to reject from the dataset"}
    )

    group_task: bool = field(
        default=True,
        metadata={"help": "Whether to group task"}
    )

    merging_method: str = field(
        default="interleave",
        metadata={"help": "The merging method"}
    )


@dataclass
class SlackParams:
    use_slack_notifier: bool = field(
        default=False,
        metadata={"help": "Whether to use slack notifier"}
    )

    slack_token: str = field(
        default="",
        metadata={"help": "The slack token"}
    )

    slack_channel: str = field(
        default="general",
        metadata={"help": "The slack channel"}
    )

    slack_message_prefix: str = field(
        default="",
        metadata={"help": "The slack message prefix"}
    )


def get_lora_model(
        model: PreTrainedModel,
        lora_config: LoraParams,
        model_type: Literal["causal", "seq2seq"] = "causal",
        print_trainable_parameters: bool = True,
):
    """
    Get LoRA model
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
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError("Please install peft library to apply PEFT LoRA (e.g. $pip install peft)")

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


def print_dataset_samples(
        tokenizer: PreTrainedTokenizer,
        dataset: Union[datasets.Dataset, datasets.IterableDataset],
        num_samples: int = 5
) -> None:
    """
    Print dataset samples
    :param tokenizer:
    :param dataset:
    :param num_samples:
    :return:
    """
    generator = iter(dataset)
    for i in range(num_samples):
        sample = next(generator)
        input_ids = sample.get("input_ids")
        output_ids = sample.get("labels")
        print("#" * 10 + f"Sample {i}" + "#" * 10)
        if input_ids is not None:
            print(f"input: {tokenizer.decode(input_ids)}")
        if output_ids is not None:
            print(f"output: {tokenizer.decode(output_ids)}")


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


def main():
    parser = HfArgumentParser(
        (ModelParams, DataParams, TrainingArguments, LoraParams, BnBParams, GenerationParams, SlackParams))
    model_args, data_args, training_args, lora_config, bnb_config, generation_args, slack_args = parser.parse_args_into_dataclasses()

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
        _sample_sp_token = list(tokenizer.special_tokens_map.values())[0]
        tokenizer.add_special_tokens({"pad_token": get_special_tokens(_sample_sp_token, "pad")})
        model = model.resize_token_embeddings(len(tokenizer))

    if bnb_config.apply_4bit_training:
        training_args.optim = "paged_adamw_32bit"

    dataset = load(
        data_name_or_path=data_args.data_name_or_path,
        data_auth_token=data_args.data_auth_token,
        streaming=data_args.streaming,
        is_supervised_dataset=data_args.is_supervised_dataset,
        group_task=data_args.group_task,
        merging_method=data_args.merging_method,
    )

    if data_args.train_split_name in dataset:
        dataset[data_args.train_split_name] = get_tokenized_dataset(
            dataset=dataset[data_args.train_split_name],
            input_column=data_args.input_column_name,
            output_column=data_args.output_column_name,
            tokenizer=tokenizer,
            model_type=model_args.model_type,
            max_input_length=data_args.max_input_length,
            max_output_length=data_args.max_output_length,
            prefix=data_args.prefix,
            suffix=data_args.suffix,
            is_train=True,
            remove_columns=True
        )

    if data_args.eval_split_name in dataset:
        dataset[data_args.eval_split_name] = get_tokenized_dataset(
            dataset=dataset[data_args.eval_split_name],
            input_column=data_args.input_column_name,
            output_column=data_args.output_column_name,
            tokenizer=tokenizer,
            model_type=model_args.model_type,
            max_input_length=data_args.max_input_length,
            max_output_length=data_args.max_output_length,
            prefix=data_args.prefix,
            suffix=data_args.suffix,
            is_train=False,
            remove_columns=True
        )

    if data_args.train_samples is not None:
        dataset[data_args.train_split_name] = dataset[data_args.train_split_name].select(range(data_args.train_samples))

    if data_args.eval_samples is not None:
        dataset[data_args.eval_split_name] = dataset[data_args.eval_split_name].select(range(data_args.eval_samples))

    trainer_callbacks = get_callbacks(
        use_slack_notifier=slack_args.use_slack_notifier,
        slack_token=slack_args.slack_token,
        slack_channel=slack_args.slack_channel,
        slack_message_prefix=slack_args.slack_message_prefix,
    )

    if data_args.train_split_name in dataset:
        print("\n\n***** Train dataset samples *****")
        print_dataset_samples(tokenizer, dataset[data_args.train_split_name], num_samples=3)

    if data_args.eval_split_name in dataset:
        print("\n\n***** Eval dataset samples *****")
        print_dataset_samples(tokenizer, dataset[data_args.eval_split_name], num_samples=3)

    data_collator = get_data_collator(model, tokenizer, model_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset[data_args.train_split_name],
        eval_dataset=dataset[data_args.eval_split_name],
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
