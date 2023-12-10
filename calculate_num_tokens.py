"""
Calculate the number of tokens in the dataset
"""
from dataclasses import dataclass, field
from multiprocessing import cpu_count

from transformers import HfArgumentParser

from dataloader.loader import load


@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    data_name_or_path: str = field(
        default="data_info_local.json",
        metadata={"help": "Path to dataset info file"},
    )

    is_supervised_dataset: bool = field(
        default=True,
        metadata={"help": "Whether the dataset is supervised or not"},
    )

    merging_method: str = field(
        default="concatenate",
        metadata={"help": "The method to merge the dataset"},
    )


def main():
    """
    Uploads a model to the HuggingFace Hub.
    1. load dataset
    2. calculate the number of tokens
    """
    parser = HfArgumentParser((DataArguments,))
    data_args = parser.parse_args_into_dataclasses()[0]

    dataset = load(
        data_name_or_path=data_args.data_name_or_path,
        streaming=False,
        is_supervised_dataset=data_args.is_supervised_dataset,
        group_task=False,
        merging_method=data_args.merging_method,
    )

    print(dataset)

    def calculate_num_tokens(example):
        example["num_tokens"] = len(example["input"].split()) + len(example["output"].split())
        return example

    dataset = dataset.map(calculate_num_tokens, num_proc=cpu_count())

    print("Total Train Tokens:", sum(dataset["train"]["num_tokens"]))
    print("Total Eval Tokens:", sum(dataset["validation"]["num_tokens"]))


if __name__ == "__main__":
    main()
