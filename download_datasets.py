"""
Downloads a dataset from the HuggingFace Hub to Local Storage.
"""
import json
import os
from dataclasses import dataclass, field
from glob import glob

from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser


def load_json(path: str) -> dict:
    """
    Loads a json file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@dataclass
class DownloadParams:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    data_info_path: str = field(
        default="data_info_local.json",
        metadata={"help": "The data info path to use"}
    )

    output_dir: str = field(
        default="data",
        metadata={"help": "The output dir to use"}
    )


def trim_name(name: str) -> str:
    """
    Trims the name to use as a directory name.
    :param name: The name to trim
    :return: The trimmed name
    """
    name = name.replace("/", "_")
    name = name.replace(":", "_")
    name = name.replace(",", "_")
    name = name.replace(".", "_")
    return name


def main():
    """
    Downloads a dataset from the HuggingFace Hub to Local Storage.
    1. load dataset
    2. save dataset to local storage
    """
    parser = HfArgumentParser((DownloadParams,))
    download_params = parser.parse_args_into_dataclasses()[0]

    data_info = load_json(download_params.data_info_path)
    for info in tqdm(data_info):
        dataset = load_dataset(
            *info["data_name_or_path"].split(","),
            use_auth_token=info["data_auth_token"]
        )

        trimmed_name = trim_name(info["data_name_or_path"])
        target_dir = os.path.join(download_params.output_dir, trimmed_name)

        os.makedirs(target_dir, exist_ok=True)
        if len(glob(os.path.join(target_dir, "*.parquet"))) > 0:
            continue

        for key, value in dataset.items():
            num_shard = (value.size_in_bytes // int(1e+8)) + 1
            for i in range(num_shard):
                file_name = f"{key}-{i:05d}-of-{num_shard:05d}.parquet"
                path = os.path.join(target_dir, file_name)
                value.shard(num_shard, i, contiguous=True).to_parquet(path)

    print("Done!")


if __name__ == "__main__":
    main()
