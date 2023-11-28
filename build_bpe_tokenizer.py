import json
import os
from dataclasses import dataclass, field
from typing import Optional

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import HfArgumentParser

from train import load


@dataclass
class DataParams:
    data_info_path: str = field(
        default="data_info_local.json",
        metadata={"help": "데이터셋 정보가 담긴 json 파일의 경로를 설정합니다."}
    )

    output_dir: str = field(
        default="tokenizer/",
        metadata={"help": "The output directory"},
    )

    vocab_size: int = field(
        default=32000,
        metadata={"help": "The size of vocabulary"},
    )

    num_proc: int = field(
        default=1,
        metadata={"help": "The number of process"},
    )

    num_examples: Optional[int] = field(
        default=None,
        metadata={"help": "The number of examples to use"},
    )


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = HfArgumentParser((DataParams,))
    data_params = parser.parse_args_into_dataclasses()[0]
    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=data_params.vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
    )

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

    tokenizer.train_from_iterator(
        _generator(),
        trainer=trainer
    )

    os.makedirs(data_params.output_dir, exist_ok=True)

    tokenizer.save(os.path.join(data_params.output_dir, "tokenizer.json"))


if __name__ == "__main__":
    main()
