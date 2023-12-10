"""
This script is used to inference with FlexFlow.
"""
import time

from dataclasses import dataclass, field
from inspect import signature

import flexflow.serve as ff
from transformers import HfArgumentParser

from utils import GenerationParams


@dataclass
class ModelParams:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    llm_name_or_path: str = field(
        default="psyche/kollama2-7b-v2",
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    ssm_name_or_path: str = field(
        default="JackFram/llama-68m",
        metadata={"help": "The model checkpoint for weights initialization."},
    )


@dataclass
class EnvParams:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    num_gpus: int = field(
        default=1,
        metadata={"help": "The number of GPUs to use."},
    )

    memory_per_gpu: int = field(
        default=80000,
        metadata={"help": "The memory per GPU to use(MiB)"},
    )

    zero_copy_memory_per_node: int = field(
        default=30000,
        metadata={"help": "The zero copy memory per node to use(MiB)"},
    )


def main(model_args, env_args, generation_args):
    """
    Uploads a model to the HuggingFace Hub.
    1. Load the model
    2. Load the SSM
    3. Compile the model
    4. Compile the SSM
    5. Inference
    """

    ff.init(
        num_gpus=env_args.num_gpus,
        memory_per_gpu=env_args.memory_per_gpu,
        zero_copy_memory_per_node=env_args.zero_copy_memory_per_node,
        tensor_parallelism_degree=env_args.num_gpus,
        pipeline_parallelism_degree=1
    )

    model = ff.LLM(model_args.llm_name_or_path)
    params = signature(ff.GenerationConfig.__init__).parameters.keys() - {"self"}
    _target_params = {}
    for p in params:
        if hasattr(generation_args, p) and getattr(generation_args, p) is not None:
            _target_params[p] = getattr(generation_args, p)

    ssms = []
    generation_config = ff.GenerationConfig(**_target_params)
    for name in model_args.ssm_name_or_path.split(","):
        if name:
            ssms.append(ff.SSM(name))
    for ssm in ssms:
        ssm.compile(generation_config)

    model.compile(generation_config, ssms=ssms)


    while True:
        input_text = input("Input: ")
        if input_text == "exit":
            break
        start = time.time()
        output = model.generate(input_text)
        print("Output:", output)
        print(f"Time:{time.time() - start:.2f} s")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelParams, EnvParams, GenerationParams))
    main(*parser.parse_args_into_dataclasses())
