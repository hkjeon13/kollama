from collections import OrderedDict
from typing import Literal

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
    >>> _get_special_tokens(tokenizer.bos_token, "sep")
    <|endoftext|>
    """
    _cat = 1 if sp_sample.startswith("<|") else (2 if sp_sample.startswith("[") else 0)
    return _SP_TOKENS_MAP[token][_cat]