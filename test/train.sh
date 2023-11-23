#!/bin/bash

DIR="$( cd "$( dirname "$0" )" && pwd -P )"

python3 $( dirname "$DIR" )/train.py \
--deepspeed=ds_configs/ds_config_zero3.json --run_name=kollama-small \
--model_name_or_path=psyche/kollama-scratch-3.5b --model_auth_token=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 \
--data_name_or_path=data_info.json --max_steps=100000 --group_task \
--save_strategy=steps --logging_strategy=steps --evaluation_strategy=no --save_steps=100 --logging_steps=100 \
--per_device_train_batch_size=6 --gradient_accumulation_steps=3 \
--add_pad_token --group_texts --is_supervised_dataset \
--do_shuffle --streaming  --do_train --output_dir=runs/