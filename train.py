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
    PreTrainedTokenizer
)

from dataloader import load
from utils import get_tokenized_dataset, GenerationParams, get_callbacks, get_data_collator, get_special_tokens
from utils.params import LoraParams, BnBParams, SlackParams


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

    num_proc: int = field(
        default=1,
        metadata={"help": "The number of processes to use"}
    )

    do_shuffle: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset"}
    )

    group_texts: bool = field(
        default=False,
        metadata={"help": "Whether to group by length"}
    )


def print_dataset_samples(
        tokenizer: PreTrainedTokenizer,
        dataset: Union[datasets.Dataset, datasets.IterableDataset],
        num_samples: int = 5
) -> None:
    """
    데이터셋의 일부 샘플을 출력
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


def main() -> None:
    """
    학습을 실행하는 메인 함수
    * 학습 과정
        1. HfArgumentParser를 이용하여 모델, 데이터, 학습 파라미터를 파싱
        2. 모델과 토크나이저를 불러옴
            - LoRA를 적용할 경우 모델에 LoRA를 적용
            - 모델에 pad 토큰을 추가할 경우 토크나이저에 pad 토큰을 추가하고 모델의 토큰 임베딩 크기를 조정
        3. 데이터셋을 불러옴
            - 데이터셋을 토크나이징
            - 데이터셋의 일부만 사용할 경우 일부만 사용
        4. Trainer를 이용하여 학습
            - 만약 학습을 실행하지 않고 평가만 실행할 경우 평가(do_train=False, do_eval=True)
    :return:

    """

    # 1. HfArgumentParser를 이용하여 모델, 데이터, 학습 파라미터를 파싱
    parser = HfArgumentParser(
        (ModelParams, DataParams, TrainingArguments, LoraParams, BnBParams, GenerationParams, SlackParams)
    )

    (model_args, data_args, training_args, lora_config,
        bnb_config, generation_args, slack_args) = parser.parse_args_into_dataclasses()

    additional_config = get_bnb_config(bnb_config)

    # 2. 모델과 토크나이저를 불러옴
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
        from utils.lora_utils import get_lora_model
        model = get_lora_model(model=model, lora_config=lora_config, model_type=model_args.model_type)

    if model_args.add_pad_token:
        _sample_sp_token = list(tokenizer.special_tokens_map.values())[0]
        tokenizer.add_special_tokens({"pad_token": get_special_tokens(_sample_sp_token, "pad")})
        model.resize_token_embeddings(len(tokenizer))
    else:
        if "eos_token" in tokenizer.special_tokens_map:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"pad token is changed to {tokenizer.eos_token}(eos token)")
        else:
            raise ValueError("pad token is not in the tokenizer")

    if bnb_config.apply_4bit_training:
        training_args.optim = "paged_adamw_32bit"

    print(model)

    # 3. 데이터셋을 불러옴
    dataset = load(
        data_name_or_path=data_args.data_name_or_path,
        data_auth_token=data_args.data_auth_token,
        streaming=data_args.streaming,
        is_supervised_dataset=data_args.is_supervised_dataset,
        group_task=data_args.group_task,
        merging_method=data_args.merging_method,
        shuffle=data_args.do_shuffle,
        num_proc=data_args.num_proc,
    )
    ## 데이터셋을 토크나이징
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
            remove_columns=True,
            group_by_length=data_args.group_texts,
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
    ## 데이터셋의 일부만 사용할 경우 일부만 사용
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

    # 4. Trainer를 이용하여 학습
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
