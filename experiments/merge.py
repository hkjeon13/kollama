import json
from glob import glob

from datasets import Dataset, load_dataset, concatenate_datasets


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base_dataset = load_dataset("psyche/instruction-gpt-3.5-turbo", token="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",
                                split="train")
    print("BASE DATASET:", base_dataset)
    files = glob("output2/*.json")
    total = []
    for file in files:
        total += load_json(file)
    dataset = Dataset.from_list(total)
    print("NEW DATASET:", dataset)
    dataset = concatenate_datasets([base_dataset, dataset])
    dataset = Dataset.from_pandas(dataset.to_pandas().drop_duplicates(ignore_index=True))
    print("MERGED DATASET:", dataset)
    dataset.push_to_hub("instruction-gpt-3.5", token="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")


if __name__ == "__main__":
    main()
