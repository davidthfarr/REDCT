""" data_utils.py

This module contains some basic helper functions for handling data for 
the LLM labeling process.

"""

from prompters.misinfo_prompter import MisinfoPrompter
from prompters.stance_prompter import StancePrompter
from prompters.humour_prompter import HumourPrompter
from prompters.ibc_prompter import IBCPrompter

def get_data_args(dataset_name):
    """
    A function that gets all the data arguments of the user provided dataset 
    name.

    Args:
        - dataset_name (str): The name of the dataset.

    Returns:
        Tuple of 
            - data_path (str): Path to local data.
            - column_map (Dict[str:str]): Dictionary that maps DF column names
                to common data schema name for each feild.
            - label_map (Dict[str:str]): Dictionary that maps the name of each
                label in a DF to an alternate naming convention. 
            - prompter (Prompter): A prompter from the ./prompters mini-library
                designed specifically for the dataset name provided.
    """
    if dataset_name == "SemEval2016" or dataset_name == "SemEval2016-test":
        column_map = {"label": "Stance", "text": "Tweet", "target": "Target"}
        label_map = {"AGAINST": "Against", "FAVOR": "For", "NONE": "Neutral"}
        prompter = StancePrompter(label_map.values(), column_map)

        # Check if train or test
        if dataset_name == "SemEval2016":
            data_path = "../data/SemEval2016/trainingdata-all-annotations.txt"
            
        else:
            data_path = "../data/SemEval2016/testdata-taskA-all-annotations.txt"

    elif dataset_name == "Misinfo" or dataset_name == "Misinfo-test":
        column_map = {"label": "gold_label", "text": "headline"}
        label_map = {"misinfo": "Misinformation", "real": "Trustworthy"}
        prompter = MisinfoPrompter(label_map.values(), column_map)

        # Check if train or test
        if dataset_name == "Misinfo":
            data_path = "../data/misinfo/train.tsv"
            
        else:
            data_path = "../data/misinfo/test.tsv"

    elif dataset_name == "humour" or dataset_name == "humour-test":
        column_map = {"label": "label", "text": "joke"}
        label_map = {0: "False", 1: "True"}
        prompter = HumourPrompter(label_map.values(), column_map)

        if dataset_name == "humour":
            data_path = "../data/humour/train.csv"

        else:
            data_path = "../data/humour/test.csv"

    elif dataset_name == "ibc" or dataset_name == "ibc-test":
        column_map = {"label": "leaning", "text": "sentence"}
        label_map = {"Conservative": "Conservative", "Neutral": "Neutral", "Liberal": "Liberal"}
        prompter = IBCPrompter(label_map.values(), column_map)

        if dataset_name == "ibc":
            data_path = "../data/IBC/train_ibc.csv"

        else:
            data_path = "../data/IBC/test_ibc.csv"

    else:
        return None, None, None, None

    return data_path, column_map, label_map, prompter

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
    meta["data"] = dataset_name 
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
    Given a text file with only the API key in it. This function reads
    the key into a string.
    """
    with open(path_to_key_file, "r") as f:
        key = f.read()
    return key

def check_gpu():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"Number of available GPUs: {device_count}")
        for i, device_name in enumerate(device_names):
            print(f"Using GPU {i}: {device_name}")
        devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
        return devices

    else:
        print("GPU is not available. Exiting Program.")
        return None
