import os
from inspect import signature
from typing import Optional, Union, Dict

from datasets import Dataset, IterableDataset
from datasets import load_from_disk, load_dataset, DatasetDict, IterableDatasetDict


def trim_dataset(
        single_data_info: Dict[str, Union[Dataset, IterableDataset, str]],
) -> Union[Dataset, IterableDataset]:
    if not isinstance(single_data_info["dataset"], (Dataset, IterableDataset)):
        raise ValueError("The dataset must be Dataset or IterableDataset")

    def transform(example):
        label = example.get(single_data_info["output_column"], None)
        example["input"] = " ".join(example[column] for column in single_data_info["input_column"])
        if label is not None:
            example["output"] = str(label)
        return example

    keys = list(next(iter(single_data_info["dataset"])).keys())

    return single_data_info["dataset"].map(transform, remove_columns=keys)


def load(
        data_name_or_path: str,
        data_auth_token: Optional[str] = None,
        streaming: bool = False,
        is_supervised_dataset: bool = False,
        group_task: bool = False,
        merging_method: str = "interleave",
        shuffle: bool = False,
        **kwargs
) -> Union[DatasetDict, IterableDatasetDict, Dataset, IterableDataset]:
    """
    Load Dataset from the Huggingface-Hub, Local Storage.
    If you set 'is_supervised_dataset' to 'True' and the 'data_name_or_path' is given as JSON file path,
    The prompts added to the dataset (The prompt is the pre-defined string).
    >>> dataset = load("nsmc", streaming=False, is_supervised_dataset=False)
    >>> dataset.__class__.__name__
    'DatasetDict'
    """
    if data_name_or_path.endswith(".json"):
        from dataloader.custom import load_datasets_from_json, SeqIO
        train_datasets = load_datasets_from_json(
            path=data_name_or_path,
            split="train",
            streaming=streaming,
            shuffle=shuffle,
        )

        eval_datasets = load_datasets_from_json(
            path=data_name_or_path,
            split="validation",
            streaming=streaming,
            shuffle=shuffle
        )

        merge_method = kwargs.get("merging_method", "interleave")
        if is_supervised_dataset:
            seqio = SeqIO(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/prompts.json"),
                tokenizer=kwargs.get("tokenizer", None)
            )

            train_datasets = seqio.transform(
                train_datasets,
                merge_method=merge_method,
                group_task=group_task,
                merging_method=merging_method,
                shuffle=shuffle,
                **kwargs
            )
            if eval_datasets:
                eval_datasets = seqio.transform(
                    eval_datasets,
                    merge_method=merge_method,
                    group_task=group_task,
                    merging_method=merging_method,
                    shuffle=shuffle,
                    **kwargs
                )
        else:
            from datasets import concatenate_datasets, interleave_datasets
            _merge_method = interleave_datasets if merge_method == "interleave" else concatenate_datasets

            train_datasets = {"train": _merge_method([trim_dataset(d) for d in train_datasets])}
            eval_datasets = _merge_method([trim_dataset(d) for d in eval_datasets]) if eval_datasets else {}

        return (DatasetDict if not streaming else IterableDatasetDict)(dict(**train_datasets, **eval_datasets))

    elif os.path.isdir(data_name_or_path):
        return load_from_disk(data_name_or_path)
    else:
        params = signature(load_dataset).parameters.keys()
        params = set(params) - {"name", "use_auth_token", "streaming"}

        return load_dataset(
            data_name_or_path,
            use_auth_token=data_auth_token,
            streaming=streaming,
            **{k: v for k, v in kwargs.items() if k in params}
        )
