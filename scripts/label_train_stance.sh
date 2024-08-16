#!/bin/bash

export OPENAI_API_TOKEN=your_open_ai_token_here!

echo "Labeling SemEval2016 dataset with GPT 3.5 and CoT Prompting"

python ./src/llm_label/openai_label.py -d SemEval2016 -o ./tests/SemEval2016_gpt -n 10 -p cot

echo "RoBERTa model with 10% confidence informed sampling expert labeling, learning with soft labels"

python ./src/edge/train.py -d ./tests/SemEval2016_gpt -o ./tests/SemEval2016_gpt -m rb -p 0.10 --conf -w

echo "Evaluating edge model on test-set"

python ./src/edge/test.py -d SemEval2016 -m ./tests/SemEval2016_gpt/roberta-base-info.json

