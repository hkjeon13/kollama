import os
import json
from tqdm import tqdm
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from datasets import load_dataset
from tokenizers import Tokenizer,decoders
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import LlamaTokenizer
from typing import Callable, Optional
from train import load


@dataclass
class DataParams:
    data_info_path: str = field(
        default="data_info.json",
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
        for d in dataset["train"]:
            yield d["input"] + " " + d["output"]

    tokenizer.train_from_iterator(
        _generator(),
        trainer=trainer
    )

    os.makedirs(data_params.output_dir, exist_ok=True)

    tokenizer.save(os.path.join(data_params.output_dir, "tokenizer.json"))
    
    tokenizer = LlamaTokenizer.from_pretrained(data_params.output_dir)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    })

    tokenizer.save_pretrained(os.path.join(data_params.output_dir, "hf"))


if __name__ == "__main__":
    main()