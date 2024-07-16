""" @file data_utils.py

@brief File that contains the data loader class for the Sem-Eval2016 dataset!
Data source: https://www.saifmohammad.com/WebPages/StanceDataset.html
Local copy: ../data/SemEval2016/..

This python file also contains some basic functions for loading the data
from the ../data/SemEval2016/ directory.

"""

import os
import json
import torch
import pickle
import scipy
import pandas as pd
import numpy as np
from collections import Counter

class DatasetLabeledByLLM(torch.utils.data.Dataset):
    """
    Basic class for working with the SemEval dataset with Pytorch.
    """

    def __init__(self, text, labels, weights, tokenizer, max_length):
        """
        Class initialized with the following:
            - text (List[str]): The text for each tweet.
            - labels (List[int]): The integer label for each class.
            - weights : The weight for each class or softlabel.
            - tokenizer : Huggingface tokenizer for the model.
            - max_length: The max length of any text provided.
        """
        self.text = text
        self.labels = labels
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        raw_text = self.text[index]

        inputs = self.tokenizer.encode_plus(
            text = raw_text,
            padding = "max_length",
            max_length = self.max_length,
            truncation = True,
            return_tensors = "pt")
        
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["attention_mask"] = inputs["attention_mask"][0]
        inputs["target"] = torch.tensor(self.labels[index], dtype=torch.int64)
        inputs["weights"] = torch.tensor(self.weights[index], dtype=torch.float32)
        
        return inputs

def load_from_LLM(path):
    """
    Function to load training data from the results from our labeled with the
    LLM.

    Args:
        - path (str): Path to experimentation folder. The folder must contain
            the metrics.pkl and results_df.pkl files created by LLM_label.py.
        - label_map (dict): Dictionary that maps string labels to ints.
    
    Returns: A dict with the following keys:
        - raw_text (List(str)): List of raw_text strings encoded as ISO-8859-1
        - true_labels (List(Int)): A list of integers which correspond to the 
            following classes to label map and contain the true labels.
        - LLM_labels (List(Int)): A list of integers which corresponds to the 
            following classes to label map and contain the LLM labels.
        - logits (List[ndarray[Float]]): A list of arrays with weights for
            each observation and each class based on logits.
    """
    data_dict = {}

    # Open the meta info to get dataset type.
    meta_path = os.path.join(path, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if meta["data"] == "SemEval2016":
        text_col = "Tweet"
        label_col = "Stance"
        label_map = {"Against": 0, "For": 1, "Neutral": 2}

    elif meta["data"] == "Misinfo":
        text_col = "headline"
        label_col = "gold_label"
        label_map = {"Misinformation": 0, "Trustworthy": 1}

    elif meta["data"] == "ideology":
        text_col = "articles"
        label_col = "label"
        label_map = {"Liberal": 0, "Conservative": 1}

    elif meta["data"] == "media":
        text_col = "text"
        label_col = "label"
        label_map = {"Left": 0, "Center": 1, "Right": 2}
                    
    elif meta["data"] == "humour":
        text_col = "joke"
        label_col = "label"
        label_map = {"False": 0, "True": 1}

    elif meta["data"] == "ibc":
        text_col = "sentence"
        label_col = "leaning"
        label_map = {"Conservative": 0, "Neutral": 1, "Liberal": 2}   
        
    elif meta["data"] == "IBC":
        text_col = "sentence"
        label_col = "leaning"
        label_map = {"Conservative": 0, "Neutral": 1, "Liberal": 2} 

    else:
        print("Oops! Invalid data from LLM found!")
        raise AttributeError

    # Open metrics
    metrics_path = os.path.join(path, "metrics.pkl")
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    # Open results df
    df_path = os.path.join(path, "results_df.pkl")
    df = pd.read_pickle(df_path)

    data_dict["logits"] = metrics["logits_clean"]
    data_dict["raw_tweets"] = df[text_col].to_list()
    data_dict["true_labels"] = df[label_col].apply(lambda x: label_map[x]).to_list()
    data_dict["LLM_labels"] = df["LLM_top_logit"].apply(lambda x: label_map[x]).to_list()

    return data_dict

def load_semeval_test():
    """ Loads the SemEval2016 test set.

    Note: The filepath is hard-coded in this instance and names of labels in 
        the test data are hard coded to align with the load_from_LLM utility
        function.

    Args: None.
    Returns:
        - raw_text (List(str)): List of raw_text strings encoded as ISO-8859-1
        - labels (List(Int)): A list of integers which correspond to the label map
    """
    file_name = "../data/SemEval2016/testdata-taskA-all-annotations.txt"
    label_map = {"AGAINST": 0, "FAVOR": 1, "NONE": 2}
    data = pd.read_table(file_name, encoding='ISO-8859-1')

    raw_tweets = data["Tweet"].to_list()
    labels = data["Stance"].apply(lambda x: label_map[x]).to_list()
    
    return raw_tweets, labels

def load_misinfo_test():
    """ Helper function to load the misinformation test data.

    Note: The filepath is hard-coded in this instance and names of labels in 
        the test data are hard coded to align with the load_from_LLM utility
        function.

    Args: None
    Returns:
    - raw_text (List(str)): List of raw_text strings encoded as ISO-8859-1
    - labels (List(Int)): A list of integers which correspond to the label map
    
    """
    file_name = "../data/misinfo/test.tsv"
    label_map = {"misinfo": 0, "real": 1}
    data = pd.read_csv(file_name, delimiter = "\t")

    raw_tweets = data["headline"].to_list()
    labels = data["gold_label"].apply(lambda x: label_map[x]).to_list()
    
    return raw_tweets, labels

def load_ideology_test():
    """ Helper function to load the ideology test data.
    
    Note: The filepath is hard-coded in this instance and names of labels in 
        the test data are hard coded to align with the load_from_LLM utility
        function.
    
    Args: None
    Returns:
    - raw_text (List(str)): List of raw_text strings encoded as ISO-8859-1
    - labels (List(Int)): A list of integers which correspond to the label map
    
    """
    file_name = "../data/ideology/test.csv"
    label_map = {"liberal": 0, "conservative": 1}
    data = pd.read_csv(file_name)
    
    raw_tweets = data["articles"].to_list()
    labels = data["label"].apply(lambda x: label_map[x]).to_list()
    
    return raw_tweets, labels

def load_media_test():
    """ Helper function to load the ideology test data.
    
    Note: The filepath is hard-coded in this instance and names of labels in 
        the test data are hard coded to align with the load_from_LLM utility
        function.
    
    Args: None
    Returns:
    - raw_text (List(str)): List of raw_text strings encoded as ISO-8859-1
    - labels (List(Int)): A list of integers which correspond to the label map
    
    """
    file_name = "../data/media/test_cleaned.csv"
    label_map = {"left": 0, "center": 1, "right": 2}
    data = pd.read_csv(file_name)
    
    raw_tweets = data["text"].to_list()
    labels = data["label"].apply(lambda x: label_map[x]).to_list()
    
    return raw_tweets, labels

def load_humour_test():
    """ Helper function to load the ideology test data.
    
    Note: The filepath is hard-coded in this instance and names of labels in 
        the test data are hard coded to align with the load_from_LLM utility
        function.
    
    Args: None
    Returns:
    - raw_text (List(str)): List of raw_text strings encoded as ISO-8859-1
    - labels (List(Int)): A list of integers which correspond to the label map
    
    """
    file_name = "../data/humour/test.csv"
    label_map = {0: 0, 1: 1}
    data = pd.read_csv(file_name)
    
    raw_tweets = data["joke"].to_list()
    labels = data["label"].apply(lambda x: label_map[x]).to_list()
    
    return raw_tweets, labels

def load_ibc_test():
    """ Loads the IBC test set.

    Note: The filepath is hard-coded in this instance and names of labels in 
        the test data are hard coded to align with the load_from_LLM utility
        function.

    Args: None.
    Returns:
        - raw_text (List(str)): List of raw_text strings encoded as ISO-8859-1
        - labels (List(Int)): A list of integers which correspond to the label map
    """
    file_name = "../data/IBC/test_ibc.csv"
    label_map = {"Conservative": 0, "Neutral": 1, "Liberal": 2}
    data = pd.read_table(file_name, delimiter =',')#, encoding='ISO-8859-1')

    raw_tweets = data["sentence"].to_list()
    labels = data["leaning"].apply(lambda x: label_map[x]).to_list()
    
    return raw_tweets, labels

# Functions helpful for simulation and handing data

def sample_training_data(data_dict, p, use_logit_conf = False, seed = None):
    """
    Function that samples the data from the data_tup to get the training data
    as tweets, labels, and logits. This handles incorporating the expert-labeled
    data.

    Args:
        - data_dict (dict): The dictionary from load_from_LLM
        - p (float): The percent of the data to randomly sample to include as 
            expert labels. If use_logit_conf is true, then this 
            is the bottom percentile to include as training data.
        - use_logits_conf (bool): Boolean indicator to sample based on logit
            confidence scores.
    Returns
        - tweets (List[str]):
        - labels (List[int]):
        - weights (List[ndarray[float]]):
    """
    rng = np.random.default_rng(seed=seed)
    raw_tweets = data_dict["raw_tweets"]
    true_labels = data_dict["true_labels"]
    LLM_labels = data_dict["LLM_labels"]
    logits = data_dict["logits"]

    labels = np.array(LLM_labels)
    _, counts = np.unique(labels, return_counts=True)
    num_classes = len(counts)

    if use_logit_conf:
        # Calculate confs and sample the bottom p percentile.
        expert_label_idx = sample_based_on_conf(logits, labels, counts, p)

    else:
        # Randomly sample the data based on p.
        n_samp = int(len(raw_tweets) * p)
        expert_label_idx = rng.choice(len(raw_tweets), size=n_samp, replace=False)

    # Assign expert labels to the LLM_labels for the sampled documents
    expert_labels = np.array(true_labels)[expert_label_idx]
    labels[expert_label_idx] = expert_labels

    # Use to expit function to transform logits
    weights = np.array([calc_weights(l) for l in logits])

    # Get expert weight vectors and reassign.
    expert_weights = np.ones((len(expert_labels), num_classes)) / 10000
    for i, index in enumerate(expert_labels):
        expert_weights[i, index] = .9998

    if len(expert_label_idx) != 0:
        weights[expert_label_idx] = expert_weights

    # Uncomment to view class balance for each run!
    #print_label_counts(labels)

    return raw_tweets, labels, weights

def sample_based_on_conf(logits, labels, counts, p):
    """
    A function that statifies the data based on the LLM labels.
    Then, it chooses expert labeled samples based on confidence score.

    Args:
        - logits (list(tuple(str, float)))
        - labels (np.ndarray(int))
        - counts (np.ndarray(int))

    Returns
        - np.ndarray of indicies sampled as expert labels
    """
    nc = len(counts)
    confs = np.array([calc_logit_conf(l) for l in logits])
    class_p = [(p / nc) / (counts[i] / labels.shape[0]) for i in range(nc)]
    class_confs = [confs[np.where(labels == idx)[0]] for idx in range(nc)]
    class_tholds = [np.quantile(class_confs[i], class_p[i]) for i in range(nc)]
    sme_label_idxs = [np.where(class_confs[i] <= class_tholds[i])[0] for i in range(nc)]
    return np.concatenate(sme_label_idxs).flatten()
    
def calc_weights(logprobs):
    """ Use expit to calculate label weights for soft labeling.

    Args:
        - logprobs (np.dnarray[float]): A vector of log-probabilites that
            correspond to each label.
    Returns: 
        np.ndarray: The vector used as soft-labels.
    """
    return scipy.special.expit(logprobs)

def calc_logit_conf(logprobs):
    """ Gets the label confidence score based on the logits.

    Note: This is n^2 due to sort but we are only sorting 2-3 classes.
    """
    probs_sorted = np.sort(logprobs)
    max_val = probs_sorted[-1]
    second_max_val = probs_sorted[-2]
    conf = np.abs(max_val - second_max_val)
    return conf

def print_label_counts(labels):
    """ Counts the counts for each label in the data.

    """
    label_counts = Counter(labels)
    
    for label, count in label_counts.items():
        print(f"Class {label}: {count} observations")
