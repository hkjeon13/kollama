from transformers import HfArgumentParser
from dataclasses import dataclass, field
import flexflow.serve as ff
from inspect import signature
from utils import GenerationParams

@dataclass
class ModelParams:
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


def main():
    parser = HfArgumentParser((ModelParams, EnvParams, GenerationParams))
    model_args, env_args, generation_args = parser.parse_args_into_dataclasses()

    ff.init(
        num_gpus=env_args.num_gpus,
        memory_per_gpu=env_args.memory_per_gpu,
        zero_copy_memory_per_node=env_args.zero_copy_memory_per_node,
    )

    model = ff.LLM(model_args.llm_name_or_path,)
    params = signature(ff.GenerationConfig.__init__).parameters.keys() - {"self"}
    _target_params = {}
    for p in params:
        if hasattr(generation_args, p):
            _target_params[p] = getattr(generation_args, p)

    ssm = ff.SSM(model_args.ssm_name_or_path)
    model.compile(generation_config=ff.GenerationConfig(**_target_params), ssms=[ssm])
    import time
    while True:
        input_text = input("Input: ")
        if input_text == "exit":
            break
        start = time.time()
        output = model.generate(input_text)
        print("Output:", output)
        print("Time:{:.2f} s".format(time.time() - start))


if __name__ == "__main__":
    main()



