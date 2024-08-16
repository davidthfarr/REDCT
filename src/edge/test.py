""" @test.py

Python module that provides testing for the edge model.

"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import RobertaModel, DistilBertTokenizer, RobertaTokenizer

from bert_classifiers import BertBasedClassifier, DistilBertClassifier
from data_utils import DatasetLabeledByLLM, check_gpu, load_test_set

def test(model, dataloader, device):
    """
    A function to evaluate the model.

    Args:
        - model (BertBaseModel): Model class (defined above).
        - dataloader (torch.utils.data.DataLoader): Testing data as DataLoader.
        - device (torch.device): A torch device used for evaluation.

    Returns: results (dict): Dictionary of testing results.
    """
    # Set model to eval mode
    model.to(device)
    model.eval()

    y_true = []
    y_hats = []
    y_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            
            logits = model(input_ids, attention_mask)
            y_hats.append(logits)
            y_preds.extend(logits.argmax(axis=1).tolist())
            y_true.extend(target.tolist())
    
    y_hats = torch.cat(y_hats)
    y_true_tensor = torch.tensor(y_true).to(device)
    
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_hats, y_true_tensor).item()

    results = {"y_true": y_true, 
               "y_hats": y_hats.tolist(),
               "y_preds": y_preds,
               "loss": loss,
               "acc": metrics.accuracy_score(y_true, y_preds),
               "f1": metrics.f1_score(y_true, y_preds, average="weighted")
               }

    return results

def get_cli_args():
    """ Gets arguments from command line
    
    """
    parser = argparse.ArgumentParser(description="Test a BERT edge model!")
    parser.add_argument('-m', type = str, required = True,
                        help = "The path to the model outfile info.json file.")
    parser.add_argument('-d', type = str, default = "SemEval2016",
                         help="Name of data to retreive testset for.")
    parser.add_argument("--max_length", type = int, default = 60, 
                        help = "Max token length of text input. Default = 60.")

    return parser.parse_args()

def main():
    """ The entry point for our program
    
    """
    args = get_cli_args()

    test_tweets, test_labels = load_test_set(args.d)

    if test_tweets is None:
        return 1

    try:
        with open(args.m, "r") as f:
            model_info = json.load(f)
            
    except FileNotFoundError:
        print(f"Failed to model info at {args.m}")
        return 1

    # Load Model
    num_classes = np.unique(np.array(test_labels)).shape[0]

    # Set logging level for transformers to only trigger on Errors.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if model_info["model_arg"] == "db":
        # Initialize for DistilBert
        model_name = "distilbert-base-cased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name,
                                                       clean_up_tokenization_spaces=True)
        model = DistilBertClassifier(num_classes)

    elif model_info["model_arg"] == "rb":
        # Initialize for RoBERTa
        model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name, truncation=True,
                                                    clean_up_tokenization_spaces=True)
        model = BertBasedClassifier(RobertaModel, model_name, num_classes)

    elif model_info["model_arg"] == "rbL":
        # Initialize for RoBERTa-L
        model_name = 'roberta-large'
        tokenizer = RobertaTokenizer.from_pretrained(model_name, truncation=True,
                                                    clean_up_tokenization_spaces=True)
        model = BertBasedClassifier(RobertaModel, model_name, num_classes)

    else:
        print("Invalid model name selected.")
        return 1

    # Create dataloader
    dummy_weights = np.zeros((len(test_tweets), 1))
    testset = DatasetLabeledByLLM(test_tweets, test_labels, dummy_weights, 
                                  tokenizer, args.max_length)
    test_loader = DataLoader(testset, batch_size = 32, shuffle = False)

    # Reload saved model
    try:
        model.load_state_dict(torch.load(model_info["model_path"]))
        print(f"Loaded model from {model_info['model_path']}.")

    except FileNotFoundError:
        print(f"Failed to model weights at {model_info['model_path']}!",
              " Check current working directory compared to relative path.")
        return 1

    # Test model!
    print("Testing Model...")
    device = check_gpu()
    results = test(model, test_loader, device)

    print(f"Test Accuracy: {results['acc']*100:.5f}%")
    print(f"Test F1: {results['f1']:.5f}")
    results_path = os.path.join(os.path.dirname(args.m), "test_results.json") 
    print(f"Saving all results to {results_path}.")

    with open(results_path, "w") as f:
        json.dump(results, f)

    return 0

if __name__ == "__main__":
    main()
