import json
import os
import random
import string
from collections import defaultdict, OrderedDict
from functools import partial
from inspect import signature
from typing import Dict, List, Union, Literal

import numpy as np
from datasets import (
    Dataset,
    IterableDataset,
    concatenate_datasets,
    interleave_datasets,
    load_dataset
)

COLUMNS = ("columns", "prompts")
AVAILABLE_DATASETS = Union[Dataset, IterableDataset]

NMT_LANGUAGE_KO = OrderedDict([
    ("ko", "한국어"), ("en", "영어"), ("ja", "일본어"), ("zh", "중국어"), ("cn", "중국어"),
    ("fr", "프랑스어"), ("de", "독일어"), ("es", "스페인어"), ("ru", "러시아어"), ("vi", "베트남어"),
    ("th", "태국어"), ("id", "인도네시아어"), ("ar", "아랍어"), ("it", "이탈리아어"), ("pt", "포르투갈어"),
])

NMT_LANGUAGE_EN = OrderedDict([
    ("en", "English"), ("ko", "Korean"), ("ja", "Japanese"), ("zh", "Chinese"), ("cn", "Chinese"),
    ("fr", "French"), ("de", "German"), ("es", "Spanish"), ("ru", "Russian"), ("vi", "Vietnamese"),
    ("th", "Thai"), ("id", "Indonesian"), ("ar", "Arabic"), ("it", "Italian"), ("pt", "Portuguese"),
])

NMT_TARGET_MAPS = {
    "ko": NMT_LANGUAGE_KO,
    "en": NMT_LANGUAGE_EN
}


def shift_tokens_right(
        input_ids: np.array,
        pad_token_id: int,
        decoder_start_token_id: int
) -> np.ndarray:
    """
    Shift input ids one token to the right.

    Args:
        input_ids (:obj:`np.ndarray`):
            The input ids.
        pad_token_id (:obj:`int`):
            The pad token id.
        decoder_start_token_id (:obj:`int`):
            The decoder start token id.

    Returns:
        :obj:`np.ndarray`: The shifted input ids.

    Examples::
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    >>> input_ids = tokenizer("Hello, my dog is cute", return_tensors="np").input_ids
    >>> shifted_input_ids = shift_tokens_right(input_ids, tokenizer.pad_token_id, tokenizer.cls_token_id)
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def span_corruption_alignment(
        tokens: List[str],
        labels: List[str],
        extra_tokens: List[str]
) -> List[str]:
    """
    Align the tokens and labels for span corruption.
    Args:
        tokens (:obj:`List[str]`):
            The tokens.
        labels (:obj:`List[str]`):
            The labels.
        extra_tokens (:obj:`List[str]`):
            The extra tokens.

    Returns:
        :obj:`List[str]`: The aligned tokens.

    TODO:
        Examples
    """

    new_tokens, n_skip, n_extra = [], 0, 0
    for token in tokens:
        if token in extra_tokens:
            new_tokens.append(token)
            n_skip += (len(labels[n_extra]) - 1)
            n_extra += 1
        else:
            if n_skip > 0:
                n_skip -= 1
            else:
                new_tokens.append(token)

    return new_tokens


class SeqIO:
    def __init__(
            self,
            prompt_file_path: str = "data/prompts.json",
            random_seed: int = 42,
            tokenizer=None
    ) -> None:

        with open(prompt_file_path, "r", encoding="utf-8") as f:
            self.prompt_list = json.load(f)

        self.rng = random.Random(random_seed)
        self.params_in_format_string = lambda x: set(s for _, s, *_ in string.Formatter().parse(x) if s is not None)
        self.tokenizer = tokenizer

    def _transform_single_dataset(
            self,
            task_type: str,
            language: str,
            dataset: AVAILABLE_DATASETS,
            mapping_table: dict,
            **kwargs
    ) -> AVAILABLE_DATASETS:
        """
        Transform a single dataset.
        :param task_type: type of task(e.g. translation, classification, mrc, etc.)
        :param language: language of dataset(e.g. ko, en, etc.)
        :param dataset: dataset to transform
        :param mapping_table: mapping table for dataset
        :param kwargs: additional arguments
        :return: transformed dataset

        >>> from datasets import load_dataset
        >>> dataset = load_dataset("wmt16", "de-en", split="train[:10]")
        >>> def preprocess_function(example):
        ...     example["source"] = example["translation"]["en"]
        ...     example["target"] = example["translation"]["de"]
        ...     example["source_language"] = "english"
        ...     example["target_language"] = "deutsch"
        ...     return example
        >>> dataset = dataset.map(preprocess_function, remove_columns=["translation"])
        >>> mapping_table =  {
        ...     "source": "source",
        ...     "target": "target",
        ...     "source_language": "source_language",
        ...     "target_language": "target_language"
        ... }
        >>> seqio = SeqIO()
        >>> target = seqio._transform_single_dataset(
        ...     task_type="translation",
        ...     language="en",
        ...     dataset=dataset,
        ...     mapping_table=mapping_table,
        ... )
        >>> target
        Dataset({
            features: ['source', 'target', 'source_language', 'target_language', 'input', 'output'],
            num_rows: 10
        })
        """
        candidates = [
            prompt["format"] for prompt in self.prompt_list
            if prompt["task"] == task_type and prompt["language"] in language.split(",")
        ]
        candidates = [
            c for c in candidates
            if len(self.params_in_format_string(" ".join(c.values())) - set(mapping_table.keys()).union(
                {"names"})) == 0
        ]
        if len(candidates) == 0:
            raise ValueError("There is no prompt for task_type: {} and language: {}".format(task_type, language))

        if task_type.endswith("classification"):
            names = dataset.features[mapping_table["category"]].names
            if names is not None:
                def process_label(example) -> dict:
                    example[mapping_table["category"]] = names[example[mapping_table["category"]]]
                    example["names"] = ", ".join(names)
                    return example

                dataset = dataset.map(process_label)
            mapping_table["names"] = "names"
        elif task_type.endswith("mrc"):
            def process_label(example) -> dict:
                ans_column = mapping_table["answer"]
                answer = example[ans_column]
                example[ans_column] = answer["text"]
                return example

            dataset = dataset.map(process_label)
        elif task_type.endswith("boolq"):
            def process_label(example) -> dict:
                ans_column = mapping_table["answer"]
                answer = example[ans_column]
                example[ans_column + "_1"] = bool(int(answer))
                return example

            dataset = dataset.map(process_label)
            dataset = dataset.remove_columns([mapping_table["answer"]])
            dataset = dataset.rename_column(mapping_table["answer"] + "_1", mapping_table["answer"])
        elif task_type.endswith("copa"):
            def process_label(example) -> dict:
                example["choices"] = " ".join(
                    [f"{i + 1}. {example[choice]}" for i, choice in enumerate(mapping_table["choices"])])
                example[mapping_table["label"] + "_1"] = str(int(example[mapping_table["label"]]) + 1)
                return example

            dataset = dataset.map(process_label)
            dataset = dataset.remove_columns([mapping_table["label"]])
            dataset = dataset.rename_column(mapping_table["label"] + "_1", mapping_table["label"])
            mapping_table["choices"] = "choices"
        elif task_type.endswith("hellaswag"):
            def process_label(example) -> dict:
                idx = int(example[mapping_table["label"]])
                example["endings"] = " ".join(
                    [f"{i + 1}. {example[ending]}" for i, ending in enumerate(mapping_table["endings"])])
                example[mapping_table["label"] + "_1"] = str(idx + 1)
                return example

            dataset = dataset.map(process_label)
            dataset = dataset.remove_columns([mapping_table["label"]])
            dataset = dataset.rename_column(mapping_table["label"] + "_1", mapping_table["label"])
            mapping_table["endings"] = "endings"
        elif task_type.endswith("multiple-choice"):
            def process_label(example) -> dict:
                ans_column = mapping_table["answer"]
                answer = example[ans_column]
                example[ans_column + "_1"] = str(int(answer) + 1)
                choices = example[mapping_table["choices"]]
                example["choices"] = ", ".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
                return example

            dataset = dataset.map(process_label)
            dataset = dataset.remove_columns([mapping_table["answer"]])
            dataset = dataset.rename_column(mapping_table["answer"] + "_1", mapping_table["answer"])
        elif task_type == "span-corruption":
            extra_tokens = self.tokenizer.additional_special_tokens \
                if hasattr(self.tokenizer, "additional_special_tokens") else None

            if len([token for token in extra_tokens if token.startswith("<extra_id")]) == 0:
                extra_tokens = [f"<extra_id_{i + 1}>" for i in range(100)]

            def process_label(examples) -> dict:
                truncated_tokens = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    return_overflowing_tokens=True,
                    add_special_tokens=False,
                    max_length=kwargs.get("max_seq_length", None),
                )
                example_output = defaultdict(list)
                sample_tokens = self.tokenizer.convert_ids_to_tokens(truncated_tokens['input_ids'][0])
                has_space_tokens = any(
                    "▁" in token for token in sample_tokens
                )

                space_separator = "▁" if has_space_tokens else ""
                span_ratio, min_interval = kwargs.get("span_ratio", 0.15), kwargs.get("min_interval", 1)
                min_span_length, max_span_length = kwargs.get("min_span_length", 1), kwargs.get("max_span_length", 10)
                for tokens in truncated_tokens['input_ids']:
                    tokens = self.tokenizer.convert_ids_to_tokens(tokens)
                    is_not_masked = [0. if "▁" not in token else 1. for token in tokens]

                    outputs, ids, initial_state = [], [], sum(is_not_masked)
                    while ((sum(len(o) for o in outputs) / len(tokens)) < span_ratio) and sum(is_not_masked) > 0:
                        span_length = self.rng.randint(min_span_length, max_span_length)
                        span_start = self.rng.choices(range(len(tokens)), k=1, weights=is_not_masked)[0]
                        span_end = span_start + int(sum(is_not_masked[span_start:(span_start + span_length)]))
                        if span_end - span_start < min_span_length:
                            continue
                        outputs.append(tokens[span_start:span_end])
                        ids.append(span_start)

                        start, end = max(0, span_start - min_interval), min(span_end + min_interval, len(is_not_masked))
                        is_not_masked[start:end] = [0. for _ in range(len(is_not_masked[start:end]))]

                    outputs = [output for _, output in sorted(zip(ids, outputs), key=lambda x: x[0])]
                    for i, start in enumerate(sorted(ids)):
                        outputs[i].insert(0, space_separator + extra_tokens[i])
                        tokens.insert(start + i, extra_tokens[i])
                    new_tokens = span_corruption_alignment(tokens=tokens, labels=outputs, extra_tokens=extra_tokens)

                    example_output[mapping_table["text"]].append(self.tokenizer.convert_tokens_to_string(new_tokens))
                    example_output[mapping_table["label"]].append(
                        " ".join([self.tokenizer.convert_tokens_to_string(output) for output in outputs]))

                return example_output

            sample = next(iter(dataset))

            dataset = dataset.map(
                process_label,
                batched=True,
                remove_columns=list(set(sample.keys()) - set(mapping_table.values()))

            )

        def example_function(example: dict) -> dict:
            io = self.rng.choice(candidates)
            return {
                "input": io["input"].format(**{
                    k: example[v] for k, v in mapping_table.items()
                    if k in self.params_in_format_string(io["input"])
                }),
                "output": io["output"].format(**{
                    k: example[v] for k, v in mapping_table.items()
                    if k in self.params_in_format_string(io["output"])
                }),
            }

        sample = next(iter(dataset))
        _rm_columns = list(set(sample.keys()) - {"input", "output"})
        params = set(signature(dataset.map).parameters.keys()) - {"self", "function", "batched", "remove_columns"}
        _additional_params = {k: v for k, v in kwargs.items() if k in params}
        dataset = dataset.map(example_function, remove_columns=_rm_columns, **_additional_params)
        return dataset

    def transform(
            self,
            datalist_with_meta: List[Dict[str, Union[str, dict, AVAILABLE_DATASETS]]],
            merge_method: Literal["interleave", "concatenate"] = "interleave",
            group_task: bool = False,
            **kwargs
    ) -> AVAILABLE_DATASETS:
        total = []
        for data in datalist_with_meta:
            total.append({
                "task_type": data["task_type"],
                "dataset": self._transform_single_dataset(
                    task_type=data["task_type"],
                    language=data["language"],
                    dataset=data["dataset"],
                    mapping_table=data["mapping_table"],
                    **data.get("kwargs", {}),
                    **kwargs
                )
            })
            data.clear()

        interleave_function = partial(interleave_datasets, stopping_strategy="all_exhausted")
        _merge_function = interleave_function if merge_method == "interleave" else concatenate_datasets

        if group_task:
            merged = defaultdict(list)
            for t in total:
                merged[t["task_type"]].append(t["dataset"])
            total = [{"task_type": k, "dataset": concatenate_datasets(v)} for k, v in merged.items()]

        return _merge_function([t["dataset"] for t in total if t["dataset"] is not None])


def translation_language_mapping(target, language_map: Dict[str, str]) -> dict:
    """
    Mapping the language code to the language name.
    :param target:
    :param language_map:
    :return:
    >>> from datasets import Dataset
    >>> dataset = Dataset.from_list([{"source_language": "en", "target_language": "ko"}])
    >>> target = {
    ...     "task_type": "translation",
    ...     "language": "ko",
    ...     "dataset": dataset,
    ...     "mapping_table": {
    ...         "source_language": "source_language",
    ...         "target_language": "target_language"
    ...     }
    ... }
    >>> language_map = {"en": "English", "ko": "Korean"}
    >>> translation_language_mapping(target, language_map)
    {'task_type': 'translation', 'language': 'ko', 'dataset': Dataset({
        features: ['source_language', 'target_language'],
        num_rows: 1
    }), 'mapping_table': {'source_language': 'source_language', 'target_language': 'target_language'}}
    """
    lang_change = lambda x: language_map.get(x, x)
    _inv_mapping = {v: k for k, v in target["mapping_table"].items()}
    target["dataset"] = target["dataset"].map(
        lambda x: {
            _inv_mapping["source_language"]: lang_change(x["source_language"]),
            _inv_mapping["target_language"]: lang_change(x["target_language"])
        }
    )
    return target


def _load_hf_dataset(data_name_or_path: str, data_auth_token: str, split="train",
                     streaming=False) -> AVAILABLE_DATASETS:
    if data_name_or_path.endswith(".txt"):
        return load_dataset("text", data_files={"train": data_name_or_path}, streaming=streaming, split=split)
    elif os.path.isdir(data_name_or_path):
        from glob import glob
        return load_dataset("text", data_files={"train": glob(os.path.join(data_name_or_path, "*.txt"))},
                            streaming=streaming, split=split)
    return load_dataset(
        *data_name_or_path.split(","),
        use_auth_token=data_auth_token,
        streaming=streaming,
        split=split
    )


def read_json(path: str) -> List[Dict[str, Union[str, dict]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_datasets_from_json(
        path: Union[str, List[Dict[str, Union[str, dict]]]],
        split: Union[Literal["train", "validation"], str] = "train",
        streaming: bool = False,
        shuffle: bool = False,
) -> List[Dict[str, Union[str, dict, AVAILABLE_DATASETS]]]:
    """
    Load datasets from JSON file.
    :param path:
    :param split:
    :param streaming:
    :param shuffle:
    :return:
    >>> data_list = [
    ...    {
    ...        "task_type": "translation",
    ...        "language": "ko",
    ...        "data_name_or_path": "psyche/nmt-sample",
    ...        "data_auth_token": None,
    ...        "train": "train",
    ...        "validation": None,
    ...        "mapping_table": {
    ...            "source": "source",
    ...            "target": "target",
    ...            "source_language": "source_language",
    ...            "target_language": "target_language"
    ...        },
    ...    }
    ... ]
    >>> load_datasets_from_json(data_list, split="train", streaming=False)
    [{'task_type': 'translation', 'language': 'ko', 'dataset': Dataset({
        features: ['source', 'target', 'source_language', 'target_language', 'category'],
        num_rows: 3
    }), 'mapping_table': {'source': 'source', 'target': 'target', 'source_language': 'source_language', 'target_language': 'target_language'}, 'kwargs': {}}]
    """

    datalist = read_json(path) if isinstance(path, str) else path
    outputs = []
    for data in datalist:
        target = None
        if data[split] is not None:
            target = {
                "task_type": data["task_type"],
                "language": data["language"],
                "dataset": _load_hf_dataset(
                    data_name_or_path=data["data_name_or_path"],
                    data_auth_token=data["data_auth_token"],
                    split=data[split],
                    streaming=streaming
                ),
                "mapping_table": data["mapping_table"],
                "kwargs": data.get("kwargs", {}),
                "input_column": data.get("input_column", ["input"]),
                "output_column": data.get("output_column", "output"),
            }
            if shuffle:
                target["dataset"] = target["dataset"].shuffle()
            if target["task_type"] == "translation" and target["language"] in NMT_TARGET_MAPS:
                target = translation_language_mapping(target, NMT_TARGET_MAPS[target["language"]])
        outputs.append(target)

    return [o for o in outputs if o is not None]
