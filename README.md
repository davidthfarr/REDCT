# RED-CT: Using LLM-labeled Data to Train and Deploy Edge Classifiers

This repository contains the code for [RED-CT: A Systems Design Methodology to Using LLM-labeled Data to Train and Deploy Edge Classifiers for Computational Social Science](./TODO_add_path).

The repo is organized as follows:
- `./data/` contains benchmark datasets.
- `./scripts/` contains examples for running the full systems methodology.
- `./src/` contains the code for research and development.
    - `./src/edge/` contains code for training and testing models for edge deployment.
    - `./src/llm_label/` contains code for using LLMs to label data.

## Methodology Flow

We provide a few command line tools to make implementing and evaluation easier. All of the code and scripts are designed to run from this top-level directory. We also provide some examples scripts in the scripts directory.

The code in this repository is designed to run on systems with cuda support!

### LLM Labeling At 'the Hub'

There are a few options for LLM data labeling.

1. `./src/llm_label/mistral_label.py` - Loads mistral7B-instruct across all available GPUs (distributed with the package accelerate) and labels the specified dataset.
2. `./src/llm_label/distribut_mistral_label.py` - Distributes the labeling process by splitting the data and loading mistral7B-instruct on each available GPU. This is the simplest (but most expensive) from of parallelization.
3. `./src/llm_label/openai_label.py` - Queries OpenAI API to label data with GPT-3.5-turbo. Other models may be supported, but the code is tested with GPT-3.5-turbo and GPT-4o.

Each of the scripts is executable with various command line arguments. Please use the `--help` or `-h` to check the CLI for each labeling method. 

```
$ python ./src/llm_label/mistral_label.py -h
$ python ./src/llm_label/distribute_mistral_label.py -h
$ python ./src/llm_label/openai_label.py -h
```

Please note that each script requires either a Huggingface token or OpenAI API token. These are accessed via environmental variables. If you are running from the terminal without using the example scripts please export the following environmental variables.

```
$ export HF_TOKEN=your_hf_token_here!
$ export OPENAI_API_TOKEN=your_openai_token_here!
```

### Classifier Training for Edge Deployment

We provide a simple CLI for finetuning various flavors of BERT models on the data labeled by the LLM labeler. Use the `-h` argument to check the CLI arguments.

```
$ python ./src/edge/train.py -h
```

### Model Evaluation and Testing

We also provide a simple CLI to evaluate the finetuned BERT model against a held-out test set. Checkout the code below to view the usage details.

```
$ python ./src/edge/test.py -h
```

### Example Scripts

- `/scripts/label_train_ibc.sh` - contains an example that distributes the labeling process onto all available GPUs. Then, we finetune distilbert with 10% confidence informed expert labeling, and evaluate the performance on a held-out test set.
- `/scripts/label_train_misinfo.sh` - contains an example for labeling with Open AIs GPT-3.5-turbo. Then, we finetune a RoBERTa-Large model naively on the labels, and evaluate the performance on a held-out test set.
- `/scripts/label_train_stance.sh` - contains an example for labeling with Open AIs GPT-4o with CoT prompting. Then, we finetune a RoBERTa model with 10% confidence informed sampling expert labeling, learning with soft labels, and evaluate the performance on a held-out test set.

Given the examples and the scripts, it should be relatively simple to reproduce all the experiments presented in the paper.

## Data

- The SemEval2016 dataset is available at https://www.saifmohammad.com/WebPages/StanceDataset.htm
- The Misinfo Reaction Frames dataset is available at https://github.com/skgabriel/mrf-modeling. 
- The humour dataset is the Reddit Jokes Database available at https://github.com/orionw/RedditHumorDetection/tree/master/data. 
- The IBC Dataset is available at https://github.com/SALT-NLP/LLMs_for_CSS/tree/main/css_data/ibc. 

Each dataset contains a collection of social media statements.
