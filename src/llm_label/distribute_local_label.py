""" @distribute_local_label.py

A CLI that provides an interface to the local_classifier. This script will
distribute the dataset to multiple GPUs. Each GPU will hold a copy of the model
and a subset of the data to label. 

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

from misinfo_prompter import MisinfoPrompter
from stance_prompter import StancePrompter
from ideology_prompter import IdeologyPrompter
from media_prompter import MediaPrompter
from humour_prompter import HumourPrompter

from local_classifier import LocalClassifier
from openai_label import read_key, read_data_to_df, get_metrics

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

# Helper functions for labeling with multiprocessing


def distribute_labeling(labeling_function, data_df, devices):
    """
    A helper function that distributes the data in df for labeling across each
    device.

    Args:
        - df (pd.DataFame): The dataframe to label.
        - labeling_function (Function): Function that takes a df and a device
            and runs inference on the given device.
        - devices (List[torch.device]): List of available GPU for inference.

    Returns:
        - df_out (pd.DataFrame): The labeled dataframe.
    """
    # Split the data into chunks for parallel processing 
    mp.set_start_method('spawn') 
    dfs = np.array_split(data_df, len(devices))  

    # Use a pool of processes to apply the function in parallel    
    with mp.Pool(processes=len(devices)) as pool:    
        df_list = pool.map(labeling_function, zip(dfs, devices))

    # Combine the results into a single DataFrame  
    df_out = pd.concat(df_list)

    return df_out

def main():
    """
    The entry point for our program!
    """
    # Ensure we have a GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"Number of available GPUs: {device_count}")
        for i, device_name in enumerate(device_names):
            print(f"Using GPU {i}: {device_name}")
        devices = [torch.device(f"cuda:{i}") for i in range(device_count)]

    else:
        print("GPU is not available. Exiting Program.")
        return 1

    # Get CLI arguments
    args = get_cli_args()

    # Ensure outpath exists
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # Load Data
    if args.d == "SemEval2016" or args.d == "SemEval2016-test":
        column_map = {"label": "Stance", "text": "Tweet", "target": "Target"}
        label_map = {"AGAINST": "Against", "FAVOR": "For", "NONE": "Neutral"}
        prompter = StancePrompter(label_map.values(), column_map)

        # Check if train or test
        if args.d == "SemEval2016":
            data_path = "../data/SemEval2016/trainingdata-all-annotations.txt"
            
        else:
            data_path = "../data/SemEval2016/testdata-taskA-all-annotations.txt"

    elif args.d == "Misinfo" or args.d == "Misinfo-test":
        column_map = {"label": "gold_label", "text": "headline"}
        label_map = {"misinfo": "Misinformation", "real": "Trustworthy"}
        prompter = MisinfoPrompter(label_map.values(), column_map)

        # Check if train or test
        if args.d == "Misinfo":
            data_path = "../data/misinfo/train.tsv"
            
        else:
            data_path = "../data/misinfo/test.tsv"

    elif args.d == "ideology" or args.d == "ideology-test":
        column_map = {"label": "label", "text": "articles"}
        label_map = {"liberal": "Liberal", "conservative": "Conservative"}
        prompter = IdeologyPrompter(label_map.values(), column_map)

        if args.d == "ideology":
            data_path = "../data/ideology/train.csv"

        else:
            data_path = "../data/ideology/test.csv"

    elif args.d == "media" or args.d == "media-test":
        column_map = {"label": "label", "text": "text"}
        label_map = {"left": "Left", "center": "Center", "right": "Right"}
        prompter = MediaPrompter(label_map.values(), column_map)

        if args.d == "media":
            data_path = "../data/media/train_cleaned.csv"

        else:
            data_path = "../data/media/test_cleaned.csv"

    elif args.d == "humour" or args.d == "humour-test":
        column_map = {"label": "label", "text": "joke"}
        label_map = {0: "False", 1: "True"}
        prompter = HumourPrompter(label_map.values(), column_map)

        if args.d == "humour":
            data_path = "../data/humour/train.csv"

        else:
            data_path = "../data/humour/test.csv"

    else:
        print("Data set invalid! Exiting Program.")
        return 1

    data_df = read_data_to_df(data_path, column_map, label_map)

    # Sub-Sample data if necessary
    if args.n > 0 and args.n <= data_df.shape[0]:
        data_df = data_df.sample(n = args.n , random_state = args.seed)


    # The model will always be Mistral
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_token = read_key("hf_secret.txt")

    # Run inference
    print("Getting Labels (this may take a few moments)...")
    start_time = time.time()
    classifier = LocalClassifier(model_name, label_map.values(), hf_token)
    
    if args.p == "zero":
        labeling_func = partial(classifier.label_df_on_device, prompt_func=prompter.simple)
        df_out = distribute_labeling(labeling_func, data_df, devices)

    elif args.p == "cot":
        labeling_func = parial(classifier.label_df_on_device, prompt_func=prompter.CoT, CoT=True)
        df_out = distribute_labeling(labeling_func, data_df, devices)

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