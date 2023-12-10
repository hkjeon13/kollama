"""
Uploads a model to the HuggingFace Hub.
"""
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser


@dataclass
class ModelParams:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    - model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models
    - use_auth_token: The token to use when uploading to huggingface.co
    - upload_name: The name of the upload
    - private: Whether to make the upload private
    """
    model_name_or_path: str = field(
        metadata={"help": "model name or path for the upload"}
    )

    use_auth_token: str = field(
        metadata={"help": "The token to use when uploading to huggingface.co"}
    )

    upload_name: str = field(
        metadata={"help": "The name of the upload"}
    )

    private: bool = field(
        default=False,
        metadata={"help": "Whether to make the upload private"}
    )


def main():
    """
    Uploads a model to the HuggingFace Hub.

    """
    parser = HfArgumentParser((ModelParams,))
    model_params = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(model_params.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_params.model_name_or_path)

    tokenizer.push_to_hub(
        model_params.upload_name,
        use_auth_token=model_params.use_auth_token,
        repo_name=model_params.upload_name,
        private=model_params.private
    )

    model.push_to_hub(
        model_params.upload_name,
        tokenizer=tokenizer,
        use_auth_token=model_params.use_auth_token,
        repo_name=model_params.upload_name,
        private=model_params.private
    )

    print("Done!")


if __name__ == "__main__":
    main()
