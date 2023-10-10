import os
from typing import Optional, Union
from inspect import signature
from datasets import load_from_disk, load_dataset, DatasetDict, IterableDatasetDict


def load(
    data_name_or_path: str,
    data_auth_token: Optional[str] = None,
    streaming: bool = False,
    is_supervised_dataset: bool = False,
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
            shuffle=kwargs.get("shuffle", False)
        )

        eval_datasets = load_datasets_from_json(
            path=data_name_or_path,
            split="validation",
            streaming=streaming,
            shuffle=False
        )
        merge_method = kwargs.get("merging_method", "interleave")
        if is_supervised_dataset:
            seqio = SeqIO(
                os.path.join(os.path.abspath(__file__), "data/prompts.json"),
                tokenizer=kwargs.get("tokenizer", None)
            )
            train_datasets = seqio.transform(train_datasets, merge_method=merge_method)
            eval_datasets = seqio.transform(eval_datasets, merge_method=merge_method)
        else:
            from datasets import concatenate_datasets, interleave_datasets

            _merge_method = interleave_datasets if merge_method == "interleave" else concatenate_datasets
            train_datasets = _merge_method(train_datasets)
            eval_datasets = _merge_method(eval_datasets)

        return (DatasetDict if not streaming else IterableDatasetDict)({
            "train": train_datasets,
            "validation": eval_datasets
        })

    elif os.path.isdir(data_name_or_path):
        return load_from_disk(data_name_or_path)
    else:
        params = signature(load_dataset).parameters.keys()
        params = set(params) - set(["name", "use_auth_token", "streaming"])

        return load_dataset(
            data_name_or_path,
            use_auth_token=data_auth_token,
            streaming=streaming,
            **{k: v for k, v in kwargs.items() if k in params}
        )

