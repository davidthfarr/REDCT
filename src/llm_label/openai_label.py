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

from data_utils import read_data_to_df, get_meta_data, get_metrics, get_logprobs

def get_cli_args():
    """
    A helper function to get the command line arguments for the script.
    """
    desc = "Use an OpenAI LLM for data labeling!"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-o', type=str,
                        help='Output directory', required=True)
    parser.add_argument('-d', type=str,
                         help='Datasets supported: "SemEval2016", "Misinfo", ' \
                              '"ibc", or "humour".', 
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

    # Get OpenAI API token 
    api_key = os.get_env("OPENAI_API_TOKEN")

    if api_key is None:
        print(f"Could not get Open AI API token environment variable! {err}.")
        return 1

    # Load data
    data_path, column_map, label_map, prompter = get_data_args(args.d)

    if data_path is None:
        print("Dataset name invalid! Exiting Program.")
        return 1

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