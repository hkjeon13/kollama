from dataclasses import dataclass, field
from inspect import signature

from transformers import HfArgumentParser, LlamaConfig, LlamaModel

from params import LlamaParams


@dataclass
class BuildParams:
    output_dir: str = field(
        default="new_model",
        metadata={"help": "The output dir to use"}
    )


def main():
    parser = HfArgumentParser((LlamaParams, BuildParams))
    model_args, build_args = parser.parse_args_into_dataclasses()
    target_params = set(signature(LlamaConfig.__init__).parameters.keys() - {"self"})
    _target_params = {key: value for key, value in model_args.__dict__.items() if key in target_params}
    config = LlamaConfig(**_target_params)
    model = LlamaModel(config)
    model.save_pretrained(build_args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
