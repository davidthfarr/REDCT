""" openai_label.py

Module that performs zero-shot data labeling using OpenAIs API.

"""

import pickle 
import json
import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score

from openai_api_interface import OpenAPI, OpenAPI_CoT
from stance_prompter import StancePrompter
from misinfo_prompter import MisinfoPrompter
from ideology_prompter import IdeologyPrompter

def read_data_to_df(path, column_map = None, label_map = None):
    """
    Helper function to read DataFrame from path.
    """
    if is_csv_path(path):
        try:
            raw_df = pd.read_csv(path)

        except UnicodeDecodeError:
            raw_df = pd.read_csv(path, encoding="ISO-8859-1")

    elif is_tsv_path(path):
        try:
            raw_df = pd.read_csv(path, delimiter = "\t")

        except UnicodeDecodeError:
            raw_df = pd.read_tsv(path, delimiter = "\t", encoding="ISO-8859-1")

    else:
        try:
            raw_df = pd.read_table(path)

        except UnicodeDecodeError:
            raw_df = pd.read_table(path, encoding="ISO-8859-1")

    # Rename labels if neccesary
    if column_map is not None and label_map is not None:
        lab_col = column_map["label"]
        raw_df[lab_col] = raw_df[lab_col].apply(lambda x: label_map[x])

    return raw_df

def get_metrics(res_df, label_logprobs, run_time, column_map):
    """
    Helper function that writes out a pkl of metrics.

    Args: 
        res_df (pd.DataFrame): The dataframe of the column map.
        label_logprobs (List[np.ndarry])
        run_time (float): The time it took to run the program.
        column_map (dict[str:str]): The column map.
    """
    metrics = {}
    lab_col = column_map["label"]

    metrics["label_acc"] = accuracy_score(res_df[lab_col], res_df["LLM_top_logit"])
    metrics["label_f1"] = f1_score(res_df[lab_col], res_df["LLM_top_logit"], average= 'weighted')
    metrics["logits_clean"] = label_logprobs
    metrics["run_time"] = run_time

    return metrics

def get_logprobs(top_logprobs, token_indicators):
    """
    Function to extract the log probs for each token indicator.

    Args:
        top_logprobs (List[Tuple()]): List of the top logits in tuple (token, logit)
        token_indicators (List[str]): Sometimes the tokenizer does not contain 
            the full token for the label. We include token_indicators to allow
            the user to specify indicators for the first token that correspond
            to labels for the LLM.

    Returns: 
        vec (ndarray): Array where n^th index corresponds to logprob of the n^th
            token in token_indicators.
    """
    vec = np.repeat(np.nan, (len(token_indicators),))
    
    token2logp = {}
    for token, logp in top_logprobs:
        if token not in token2logp:
            token2logp[token] = logp

    min_logprob = min(token2logp.values()) - 1

    for idx, token in enumerate(token_indicators):
            vec[idx] = token2logp.get(token, min_logprob)

    return vec

def get_meta_data(args):
    """
    Helper function to get metadata from the argument parser.
    """
    meta = {}
    meta["data"] = args.d 
    meta["prompt_type"] = args.p 
    meta["n"] = args.n
    meta["seed"] = args.seed
    meta["model"] = args.m

    return meta

def is_csv_path(path):
    """
    Helper function to see if path has csv handle.
    """
    return True if path[-4:] == ".csv" else False

def is_tsv_path(path):
    """
    Helper function to see if path has tsv handle.
    """
    return True if path[-4:] == ".tsv" else False

def read_key(path_to_key_file):
    """
    Given a text file with only the OpenAI API key in it. This function reads
    the key into a string.
    """
    with open(path_to_key_file, "r") as f:
        key = f.read()
    return key

def get_cli_args():
    """
    A helper function to get the command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Use an LLM for Stance Classification!")
    parser.add_argument('-o', type=str,
                        help='Output directory', required=True)
    parser.add_argument('-d', type=str,
                         help='Datasets supported: "SemEval2016", "Misinfo", "ideology".', 
                         required=True)
    parser.add_argument('-p', type=str,
                        help= 'Prompting type: "zero" or "few"',
                        required=False, default="zero")
    parser.add_argument('-n', type=int,
                        help= 'Number of documents to sample to classify stance' \
                            "Default set to 10. To run with all set to -1.",
                        required=False, default=10)
    parser.add_argument('-m', type=str,
                        help= 'OpenAI model to use for classification.',
                        required=False, default="gpt-3.5-turbo")
    parser.add_argument('--seed', type = int,
                        help="The random state or seed for sampling",
                        required=False, default=None)
    parser.add_argument("--logit_bias", type=int, default=10, required = False,
                        help="To implement logit bias in the OpenAI API call")
    return parser.parse_args()

def main():
    """
    The entry point for the program!
    """
    args = get_cli_args()
    
    # Ensure outpath exists
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # Read API key
    api_key = read_key("secret.txt")

    # Get data information
    if args.d == "SemEval2016":
        data_path = "../data/SemEval2016/trainingdata-all-annotations.txt"
        column_map = {"label": "Stance", "text": "Tweet", "target": "Target"}
        label_map = {"AGAINST": "Against", "FAVOR": "For", "NONE": "Neutral"}
        prompter = StancePrompter(label_map.values(), column_map)

    elif args.d == "Misinfo":
        data_path = "../data/misinfo/train.tsv"
        column_map = {"label": "gold_label", "text": "headline"}
        label_map = {"misinfo": "Misinformation", "real": "Trustworthy"}
        prompter = MisinfoPrompter(label_map.values(), column_map)

    elif args.d == "ideology":
        data_path = "../data/ideology/train.csv"
        column_map = {"label": "label", "text": "articles"}
        label_map = {"liberal": "Liberal", "conservative": "Conservative"}
        prompter = IdeologyPrompter(label_map.values(), column_map)

    else:
        print(f"Data set invalid: {args.d}!")
        return 1

    # Load data
    data_df = read_data_to_df(data_path, column_map, label_map)

    # Sub-Sample data if necessary
    if args.n > 0 and args.n <= data_df.shape[0]:
        data_df = data_df.sample(n = args.n , random_state = args.seed)

    # Init API and get prompts based on classification type
    if args.p == "zero":
        api = OpenAPI(api_key, args.m)
        prompt_list = data_df.apply(lambda row: 
                                    prompter.simple(**row), 
                                    axis = 1).tolist()

    elif args.p == "cot":
        api = OpenAPI_CoT(api_key, args.m)
        prompt_list = prompt_list = data_df.apply(lambda row: 
                                                  prompter.CoT(**row),
                                                  axis = 1).tolist()
    
    else:
        print("Prompting type invalid!")
        return 1

    # Get token IDs for logit bias and cleaning up top logit column
    label_token_ids = api.get_token_IDs(label_map.values())
    logit_bias_dict = {tok[0]: args.logit_bias for tok in label_token_ids}
    token_indicators = api.decode_token_IDs(logit_bias_dict.keys())

    # Get labels!
    print("Getting Labels (this may take a few moments)...")
    start_time = time.time()
    api.request_prompt(prompt_list, logit_bias = logit_bias_dict)
    
    # Parse responses
    parsed_responses = api.parse_responses()
    run_time = time.time() - start_time

    # Add to basic info to data_df
    data_df["prompt"] = prompt_list
    data_df["LLM_raw_label"] = parsed_responses["Label"]
    data_df["logits"] = parsed_responses["Logits"]

    # Get top logit by taking the max indicator token probability
    label_logprobs = [get_logprobs(l, token_indicators) for l in data_df["logits"]]
    labels_list = list(label_map.values())
    data_df["LLM_top_logit"] = [labels_list[logps.argmax()] for logps in label_logprobs]
    
    if args.p == "cot":
        data_df["CoT"] = parsed_responses["CoT"]

    # Write DF and metrics to pickle
    df_path = os.path.join(args.o, "results_df.pkl")
    print(f"Writing dataframe to {df_path}...")
    data_df.to_pickle(df_path)
    
    metrics = get_metrics(data_df, label_logprobs, run_time, column_map)
    metrics_path = os.path.join(args.o, "metrics.pkl")
    print(f"Writing metrics dictionary to {metrics_path}...")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    # Save metrics and meta data
    meta = get_meta_data(args)
    meta_path = os.path.join(args.o, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    print("Complete!")
    return 0

if __name__ == "__main__":
    main()