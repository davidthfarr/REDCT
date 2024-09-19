#!/bin/bash

export HF_TOKEN=your_hf_token_here!

echo "Labeling SemEval2016 dataset with Llama3.18B"

python ./src/llm_label/llama_label.py -d SemEval2016 -o ./tests/SemEval2016_llama -n 10

echo "RoBERTa model with 10% confidence informed sampling expert labeling, learning with soft labels"

python ./src/edge/train.py -d ./tests/SemEval2016_gpt -o ./tests/SemEval2016_gpt -m rb -p 0.10 --conf -w

echo "Evaluating edge model on test-set"

python ./src/edge/test.py -d SemEval2016 -m ./tests/SemEval2016_gpt/roberta-base-info.json

