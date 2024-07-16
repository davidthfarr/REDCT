""" @train.py

A python module that trains or fine-tunes the edge model.

"""

import os
import sys
import json
import argparse
import time
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaModel, DistilBertTokenizer, RobertaTokenizer, get_scheduler

from bert_classifiers import BertBasedClassifier, DistilBertClassifier
from data_utils import DatasetLabeledByLLM, load_from_LLM, sample_training_data

def train(model, dataloader, n_epochs, optimizer, device, use_weights = False):
    """
    Function that trains the model. 
    Args:
        - model (BertBaseModel): Model class (defined above).
        - dataloader (torch.utils.data.DataLoader): Training data as DataLoader
        - n_epochs (int): Number of training epochs
        - optimizer (torch.optim): Optimizer initialized with model params.
        - accelerator (Accelerator()): A accelerate device for dist training.
        - use_weights (bool): Boolean indicator to include sample weights in the
            training process.

    Returns: logs (dict): A dictionary with training logs per epoch.
    """
    # Set model in training mode
    model.to(device)
    model.train()
    
    num_training_steps = n_epochs * len(dataloader)
    lr_scheduler = get_scheduler("linear", 
                                 optimizer=optimizer, 
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    
    logs = {"epoch": [], "loss": [], "train_acc": []}
    progress_bar = tqdm(range(num_training_steps))

    start_time = time.time()
    for e in range(n_epochs):
        n_correct = 0
        n_total = 0
        running_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            weights = batch["weights"].to(device)

            y_hats = model(input_ids, attention_mask)
            y_preds = y_hats.argmax(axis = 1)
            n_total += y_preds.shape[0]
            n_correct += (target ==  y_preds).sum().item()
            n_total
            
            loss_func = torch.nn.CrossEntropyLoss()
            if use_weights:
                # Add weighted code here
                loss = loss_func(y_hats, weights) 

            else:
                loss = loss_func(y_hats, target)
        
            running_loss += loss.item()

            # Zero out grads, calc grads w.r.t. loss, perform backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

        logs["epoch"].append(e)
        logs["loss"].append(running_loss)
        logs["train_acc"].append(n_correct/n_total)
        pbar_msg = f"epoch: {e+1}, loss: {running_loss:.3f}, acc: {n_correct/n_total:.3f}"
        progress_bar.set_postfix_str(pbar_msg)
        running_loss, n_total, n_correct = 0, 0, 0

    train_time = time.time() - start_time
    logs["train_time"] = train_time

    return logs

def check_gpu():
    """
    A helper function to check for available GPUs.
    Returns: device (torch.device): A torch device leveraged to
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Number of available GPUs: {device_count}")
        print(f"Using GPU {current_device}: {device_name}")
        device = torch.device("cuda")

    else:
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")

    return device

def get_cli_args():
    """ Retreives commandline arguments from user.
    
    """
    description = "Train a BERT-based edge model based on LLM labeled data!"
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('-d', type = str, required = True,
                         help="Path of training data. Must contains files: " \
                               "'metrics.pkl' and 'results_df.pkl'.")
    
    parser.add_argument('-o', type = str, required = True,
                        help='Directory to save model train logs and model to.')
    
    parser.add_argument('-m', type = str, required = False, default = "db",
                        help = "Name of the model. Default is 'db'." \
                               "Available options are: 'db' for DistilBERT, " \
                               "'rb' for RoBERTa, and 'rbL' for RoBERTa-L. " \
                               "(default: db)")

    parser.add_argument('-w', action = argparse.BooleanOptionalAction,
                        help = "Include flag to train using weighted loss or " \
                        "soft labels.")

    parser.add_argument('-p', type = float, required = False, default = 0.0,
                        help = "The percent of the data randomly selected as " \
                            "expert labels. With '--conf' flag it specifies " \
                            "the bottom percentile to include. (default: 0.0)")

    parser.add_argument('--conf', action = argparse.BooleanOptionalAction,
                        help = "Specify this flag to use the confidence scores " \
                                "when selecting documents for expert labels")

    parser.add_argument("--max_length", type = int, default = 60, 
                        help = "Max token length of text input. (default: 60)")

    parser.add_argument("--n_epochs", type = int, default = 10, 
                        help= "Number of training epochs. (default: 10)")

    parser.add_argument("--batch_size", type = int, default = 32, 
                        help= "Batch size for training. (default: 32)")

    parser.add_argument("--learning_rate", type = float, default = 1e-6,
                        help="Learning rate for optimizer. (default: 1e-6)")

    return parser.parse_args()

def main():
    """ The entry point for the program!
    
    """
    # Get CLI Args
    args = get_cli_args()

    # Ensure out-path exists
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # Load data
    try:
        data_dict = load_from_LLM(args.d)

    except FileNotFoundError:
        print(f"{args.d} is an invalid data directory.")
        return 1

    samp = sample_training_data(data_dict, p = args.p, use_logit_conf = args.conf)
    tweets, labels, weights = samp
    num_classes = np.unique(labels).shape[0]

    device = check_gpu()
    
    if device == torch.device("cpu"):
        return 1

    # Set logging level for transformers to only trigger on Errors.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if args.m == "db":
        # Initialize for DistilBert
        model_name = "distilbert-base-cased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertClassifier(num_classes)

    elif args.m == "rb":
        # Initialize for RoBERTa
        model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name, truncation=True)
        model = BertBasedClassifier(RobertaModel, model_name, num_classes)

    elif args.m == "rbL":
        # Initialize for RoBERTa-L
        model_name = 'roberta-large'
        tokenizer = RobertaTokenizer.from_pretrained(model_name, truncation=True)
        model = BertBasedClassifier(RobertaModel, model_name, num_classes)

    else:
        print("Invalid model name selected.")
        return 1

    print(f"Loaded base model: {model_name}")

    # Assign model to GPU if available
    trainset = DatasetLabeledByLLM(tweets, labels, weights, tokenizer, args.max_length)
    train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    print("Training Model...")
    logs = train(model, train_loader, args.n_epochs, optimizer, device, args.w)

    # Save model and info
    model_path = os.path.join(args.o, (model_name + ".pt"))
    torch.save(model.state_dict(), model_path)

    logs["model_path"] = model_path
    logs["training_data"] = args.d
    logs["model_arg"] = args.m 
    logs["used_weights"] = args.w
    logs["p_arg"] = args.p
    logs["cong_arg"] = args.conf
    logs["max_length"] = args.max_length
    logs["batch_size"] =  args.batch_size

    logs_path = os.path.join(args.o, (model_name + "-info.json"))
    with open(logs_path, "w") as f:
        json.dump(logs, f)

    print(f"Model saved to {model_path}. Model info saved to {logs_path}.")
    return 0

if __name__ == "__main__":
    main()
