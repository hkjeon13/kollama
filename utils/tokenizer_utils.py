from collections import OrderedDict
from typing import Literal
from typing import Optional, Union, List

import datasets
from transformers import PreTrainedTokenizer

_SP_TOKENS_MAP = OrderedDict([
    ("bos", ["<s>", "<|startoftext|>", "[CLS]"]),
    ("eos", ["</s>", "<|endoftext|>", "[SEP]"]),
    ("sep", ["<sep>", "<|sep|>", "[SEP]"]),
    ("pad", ["<pad>", "<|pad|>", "[PAD]"]),
    ("unk", ["<unk>", "<|unk|>", "[UNK]"]),
    ("mask", ["<mask>", "<|mask|>", "[MASK]"]),
])


def get_special_tokens(
        sp_sample: str,
        token: Literal["bos", "eos", "sep", "pad", "unk", "mask"]
) -> str:
    """
    sp_sample: str
    token: Literal["bos", "eos", "sep", "pad", "unk", "mask"]
    returns:
        str of special token

    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> get_special_tokens(tokenizer.bos_token, "sep")
    <|endoftext|>
    """
    _cat = 1 if sp_sample.startswith("<|") else (2 if sp_sample.startswith("[") else 0)
    return _SP_TOKENS_MAP[token][_cat]


def get_tokenized_dataset(
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        model_type: Literal["causal", "seq2seq"],
        input_column: str,
        output_column: str,
        max_input_length: int = 512,
        max_output_length: int = 512,
        prefix: str = "",
        suffix: str = "",
        is_train: bool = True,
        remove_columns: Optional[Union[List[str], bool]] = None,
) -> datasets.Dataset:
    """
    모델의 유형에 따라 입력과 출력을 토큰화하여 반환
    :param dataset: 타깃 데이터셋
    :param tokenizer: 토크나이저
    :param model_type: 모델의 유형
    :param input_column: 입력 컬럼
    :param output_column: 출력 컬럼
    :param max_input_length: 입력 최대 길이
    :param max_output_length: 출력 최대 길이
    :param prefix: 입력 앞에 붙일 토큰
    :param suffix: 입력 뒤에 붙일 토큰
    :param is_train: 훈련 데이터인지 여부
    :param remove_columns: 제거할 컬럼
    :return:
        - 토큰화된 데이터셋

    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> dataset = datasets.load_dataset("squad_kor_v1")
    >>> dataset = get_tokenized_dataset(
    ...     dataset=dataset["train"],
    ...     tokenizer=tokenizer,
    ...     model_type="causal",
    ...     input_column="question",
    ...     output_column="context",
    ...     max_input_length=512,
    ...     max_output_length=512,
    ...     prefix="질문: ",
    ...     suffix=" | 답변: ",
    ...     is_train=True,
    ...     remove_columns=["id", "title", "answers", "context_id", "question_id"]
    ... )
    """
    if isinstance(remove_columns, bool):
        keys = list(next(iter(dataset)).keys())
        remove_columns = keys if remove_columns else []

    def tokenize_function(examples):
        inputs, outputs = [], []
        examples[input_column] = [prefix + input_text + suffix for input_text in examples[input_column]]
        inputs.append(examples[input_column])
        outputs.append(examples[output_column])

        if model_type == "causal" and is_train:
            inputs.append(examples[output_column])

        if model_type == "causal" and not is_train:
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
        else:
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
        tokenized_inputs = tokenizer(*inputs, padding="max_length", truncation=True, max_length=max_input_length)

        if model_type == "causal" and not is_train:
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"

        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy() if model_type == "causal" and is_train \
            else tokenizer(*outputs, padding="max_length", truncation=True, max_length=max_output_length)["input_ids"]

        return tokenized_inputs

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns
    )
