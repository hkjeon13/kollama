#!/bin/bash

DIR="$( cd "$( dirname "$0" )" && pwd -P )"

python3 $( dirname "$DIR" )/train.py \
--model_name_or_path=psyche/kogpt \
--model_type=causal \
--data_name_or_path=data/data_info.json \
--max_input_length=512 \
--max_output_length=512 \
--max_new_tokens=512 \
--train_samples=100000 \
--per_device_train_batch_size=1 \
--output_dir=$5 \
--max_steps=1 \
--num_train_epochs=1 \
--add_pad_token \
--do_train \
--streaming