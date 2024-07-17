""" @local_label.py

A CLI that provides an interface to the local_classifier. This script will label
documents using Mistral7B-Instruct.


"""

import argparse
import sys
import os
import pickle
import time
import json
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

from prompters.misinfo_prompter import MisinfoPrompter
from prompters.stance_prompter import StancePrompter
from prompters.humour_prompter import HumourPrompter
from prompters.ibc_prompter import IBCPrompter

from local_classifier import LocalClassifier
from data_utils import read_data_to_df, get_meta_data, get_metrics, get_data_args, check_gpu

def get_cli_args():
    """
    A helper function to get the command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Use a local LLM for Stance Classification!")
    parser.add_argument('-o', type=str,
                        help='Output directory', required=True)
    parser.add_argument('-d', type=str,
                         help='Dataset name: "SemEval2016" or "SemEval2016-test".', 
                         required = False, default = "SemEval2016")
    parser.add_argument('-p', type=str,
                        help='Prompting type: "zero", "cot"',
                        required=False, default="zero")
    parser.add_argument('-n', type=int, default=10,
                       help='The number of samples to draw for classification.')
    parser.add_argument('--seed', type=int, default=None,
                       help="Random seed used for sampling")
    return parser.parse_args()

def main():
    """
    The entry point for our program!
    """
    # Check for GPUs
    devices = check_gpu()

    if devices is None:
        print("GPU is not available. Exiting Program.")
        return 1

    # Get CLI arguments
    args = get_cli_args()

    # Get HF token
    hf_token = read_key("hf_secret.txt")

    # Ensure outpath exists
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    data_path, column_map, label_map, prompter = get_data_args(args.d)

    if data_path is None:
        print("Dataset name invalid! Exiting Program.")
        return 1

    data_df = read_data_to_df(data_path, column_map, label_map)

    # Sub-Sample data if necessary
    if args.n > 0 and args.n <= data_df.shape[0]:
        data_df = data_df.sample(n = args.n , random_state = args.seed)

    # Run inference
    print("Getting Labels (this may take a few moments)...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    start_time = time.time()
    classifier = LocalClassifier(model_name, label_map.values(), hf_token)
    
    if args.p == "zero":
        df_out = classifier.label_df_on_device(df_device_tuple=(data_df, "auto"),
                                               prompt_func=prompter.simple)

    elif args.p == "cot":
        df_out = classifier.label_df_on_device(df_device_tuple=(data_df, "auto"),
                                               prompt_func=prompter.CoT,
                                               cot=True)

    else:
        print("Invalid prompting type")
        return 1

    run_time = time.time() - start_time

    # Write DF and metrics to pickle
    df_path = os.path.join(args.o, "results_df.pkl")
    print(f"Writing dataframe to {df_path}...")
    df_out.to_pickle(df_path)

    metrics = get_metrics(df_out, df_out["logits_clean"], run_time, column_map)
    met_path = os.path.join(args.o, "metrics.pkl")
    print(f"Writing metrics dictionary to {met_path}...")
    with open(met_path, 'wb') as f:
        pickle.dump(metrics, f)

    # Save meta data
    meta = {}
    meta["data"] = args.d 
    meta["prompt_type"] = args.p 
    meta["model"] = "mistral"
    meta_path = os.path.join(args.o, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    print("Complete!")
    return 0

if __name__ == "__main__":
    main()