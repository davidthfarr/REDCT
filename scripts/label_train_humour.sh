#!/bin/bash

export HF_TOKEN=your_hf_token_here!

echo "Labeling the humour dataset with Mistral 7B"

python ./src/llm_label/mistral_label.py -d humour -o ./tests/humour_mistral -n 100

echo "Finetuning distilbert with 50% randomly sampled expert labels"

python ./src/edge/train.py -d ./tests/humour_mistral -o ./tests/humour_mistral -p 0.50

echo "Evaluating edge model on test-set"

python ./src/edge/test.py -d humour -m ./tests/humour_mistral/distilbert-base-cased-info.json

