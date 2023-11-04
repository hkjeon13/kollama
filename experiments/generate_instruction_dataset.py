import json
import os
from dataclasses import dataclass, field

import openai
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser


@dataclass
class APIParams:
    token_file: str = field(
        metadata={"help": "The token file for accessing model files on huggingface.co/models"},
    )

    api_model_name: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "The model name to use"},
    )


@dataclass
class DataParams:
    data_name_or_path: str = field(
        metadata={"help": "Path to dataset or dataset identifier from huggingface.co/datasets"},
    )

    data_auth_token: str = field(
        default="",
        metadata={"help": "The auth token for accessing dataset files on huggingface.co/datasets"},
    )

    target_split_name: str = field(
        default="train",
        metadata={"help": "The name of the target split"},
    )

    input_column_name: str = field(
        default="input",
        metadata={"help": "The column to choose from the dataset"},
    )
    start_index: int = field(
        default=0,
        metadata={"help": "The start index of the dataset"},
    )


@dataclass
class SaveParams:
    output_dir: str = field(
        metadata={"help": "The output directory"},
    )

    save_name: str = field(
        default="output",
        metadata={"help": "The name of the saved model"},
    )

    save_steps: int = field(
        default=100,
        metadata={"help": "The number of steps to save"},
    )


def get_answer_from_question(question: str, model_name: str = "gpt-3.5-turbo"):
    output = {"question": question, model_name: ""}
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": output["question"]}],
        )
        output[model_name] = completion.choices[0].message.content
    except Exception as e:
        print(e)
        print("OpenAI API Error")
    return output


def main():
    parser = HfArgumentParser((APIParams, DataParams, SaveParams))
    api_params, data_params, save_params = parser.parse_args_into_dataclasses()
    with open(api_params.token_file, "r") as f:
        content = json.load(f)

    if isinstance(content, dict):
        openai.api_key = content.get("token", KeyError("api_key not found"))

    dataset = load_dataset(
        data_params.data_name_or_path,
        use_auth_token=data_params.data_auth_token,
        split=data_params.target_split_name,
    )

    outputs = []
    os.makedirs(save_params.output_dir, exist_ok=True)
    save_steps = 1
    for i, example in tqdm(enumerate(dataset)):
        if i < data_params.start_index:
            continue
        model_output = get_answer_from_question(example[data_params.input_column_name])
        if model_output[api_params.api_model_name]:
            outputs.append(model_output)
        if len(outputs) % save_params.save_steps == 0 and len(outputs) > 0:
            with open(os.path.join(save_params.output_dir, f"{save_params.save_name}_{save_steps}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)
            save_steps += 1
            outputs = []

    if len(outputs) > 0:
        with open(os.path.join(save_params.output_dir, f"{save_params.save_name}_{save_steps}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
