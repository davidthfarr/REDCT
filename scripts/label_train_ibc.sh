#!/bin/bash

export HF_TOKEN=your_hf_token_here!

echo "Labeling the IBC dataset with Mistral 7B and all available GPUs"

python ./src/llm_label/distribute_mistral_label.py -d ibc -o ./tests/ibc_mistral -n 100

echo "Finetuning distilbert with 10% confidence informed expert labeling"

python ./src/edge/train.py -d ./tests/ibc_mistral -o ./tests/ibc_mistral -p 0.10 --conf

echo "Evaluating edge model on test-set"

python ./src/edge/test.py -d ibc -m ./tests/ibc_mistral/distilbert-base-cased-info.json

