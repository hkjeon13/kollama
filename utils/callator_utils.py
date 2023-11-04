from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq


def get_data_collator(model, tokenizer, model_args):
    """
    Get data collator
    :param model:
    :param tokenizer:
    :param model_args:
    :return:

    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> tokenizer = AutoTokenizer.from_pretrained("psyche/kogpt")
    >>> model = AutoModelForCausalLM.from_pretrained("psyche/kogpt")
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class ModelParams:
    ...     model_type: str = "causal"
    >>> model_args = ModelParams()
    >>> get_data_collator(model, tokenizer, model_args).__class__.__name__
    'DataCollatorForLanguageModeling'
    """
    from inspect import signature

    collator_module = DataCollatorForLanguageModeling \
        if model_args.model_type == "causal" else DataCollatorForSeq2Seq
    candidates = {
        "tokenizer": tokenizer, "model": model, "mlm": False,
        "pad_to_multiple_of": None
    }
    para_names = signature(collator_module.__init__).parameters.keys()
    collator_args = {
        key: candidates.get(key) for key in para_names if key in candidates
    }

    return collator_module(**collator_args)
