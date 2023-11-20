from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class ModelParams(object):
    model_name_or_path: str = field(
        metadata={"help": "model name or path for the upload"}
    )

    use_auth_token: str = field(
        metadata={"help": "The token to use when uploading to huggingface.co"}
    )

    upload_name: str = field(
        metadata={"help": "The name of the upload"}
    )


def main():
    parser = HfArgumentParser((ModelParams,))
    model_params = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(model_params.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_params.model_name_or_path)

    model.push_to_hub(
        model_params.upload_name,
        tokenizer=tokenizer,
        use_auth_token=model_params.use_auth_token,
        repo_name=model_params.upload_name
    )

    print("Done!")


if __name__ == "__main__":
    main()