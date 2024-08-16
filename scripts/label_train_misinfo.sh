#!/bin/bash

export OPENAI_API_TOKEN=your_openai_token_here!

echo "Labeling the Misinfo dataset with GPT-3.5"

python ./src/llm_label/openai_label.py -d Misinfo -o ./tests/Misinfo_gpt

echo "Finetuning RoBERTa-Large model naivley on the labels"

python ./src/edge/train.py -d ./tests/Misinfo_gpt -o ./tests/Misinfo_gpt -m rbL

echo "Evaluating edge model on test-set"

python ./src/edge/test.py -d Misinfo -m ./tests/Misinfo_gpt/roberta-large-info.json


