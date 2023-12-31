"""
Script to build a SentencePiece tokenizer from a dataset.
"""
from dataclasses import dataclass, field
from typing import Optional

import sentencepiece as spm
from transformers import HfArgumentParser

from train import load


@dataclass
class BuildingParams:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_prefix: str = field(
        default="llama",
        metadata={"help": "The output directory"},
    )

    model_type: str = field(
        default="bpe",
        metadata={"help": "The number of process"},
    )

    vocab_size: int = field(
        default=32000,
        metadata={"help": "The size of vocabulary"},
    )


@dataclass
class DataPrams:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    data_info_path: str = field(
        default="data_info_local.json",
        metadata={"help": "데이터셋 정보가 담긴 json 파일의 경로를 설정합니다."}
    )

    num_proc: int = field(
        default=1,
        metadata={"help": "The number of process"},
    )

    num_examples: Optional[int] = field(
        default=None,
        metadata={"help": "The number of examples to use"},
    )


def main(building_params, data_params):
    """
    Uploads a model to the HuggingFace Hub.
    1. load dataset
    2. train tokenizer
    """

    dataset = load(
        data_name_or_path=data_params.data_info_path,
        data_auth_token=None,
        streaming=True,
        is_supervised_dataset=True,
        group_task=True,
        merging_method="interleave",
        shuffle=True,
        num_proc=data_params.num_proc,
    )

    def _generator() -> str:
        loader = dataset["train"]
        if data_params.num_examples is not None:
            loader = loader.take(data_params.num_examples)
        for d in loader:
            yield d["input"] + " " + d["output"]

    spm.SentencePieceTrainer.train(
        sentence_iterator=_generator(),
        model_prefix=building_params.model_prefix,
        vocab_size=building_params.vocab_size,
        model_type=building_params.model_type
    )

    print(f"Model trained and saved with prefix '{building_params.model_prefix}'")


if __name__ == "__main__":
    parser = HfArgumentParser((BuildingParams, DataPrams))
    main(*parser.parse_args_into_dataclasses())
