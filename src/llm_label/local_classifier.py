""" @local_classifier.py

Class definition of a LLM-based classifier. This module is designed to work with 
mistral's open model at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2.
However, other models used for text-generation could work with the 
class LocalStanceClassifier.

"""

from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import get_logprobs

class LocalClassifier:
    """
    A class that contains the basic logic for using Local LLMs as classifiers.
    for zero-shot or few-shot classification tasks. This is designed to run on
    one GPU!

    Class attributes:
        Initialized with:
        - model_name (str): The name of the model/tokenizer.
        - label_indicators (dict): token label indicators map to class labels.
        - hf_token (str): The hugging face token.
        - device (torch.device | str): The device to load and run the model on.
        Defined on init:
        - model (): The huggingface model.
        - tokenizer (): The hugging face tokenizer.

    Public class methods:
        - 
    """

    def __init__(self, model_name, labels, hf_token = None):
        """
        Initializes the LocalClassifer
        """
        # Defined class attributes
        self.model_name = model_name
        self.hf_token = hf_token
        self.labels = list(labels)

    def label_df_on_device(self, df_device_tuple, prompt_func, top_logprobs = 50, 
                           CoT = False, max_response_len = 200):
        """
        Public method to label a df on a device.

        Args:
            - df_device_tuple tuple(pd.DataFrame:torch.device): 
                A tuple that contains the dataset to label and the device to
                to store the model for.
            - prompt_func (Callable()): Function that takes an observation and 
                target and creates a prompt (see prompting.py).
            - top_logprobs (int): The number of top logits to return
            - CoT (bool): Boolean indicator to perform CoT prompting.
            - max_response_len (int): The maximum number of tokens to return for
                CoT response if CoT is True.
            - progbar (bool): Boolean indicator to include progress bar
                for labeling. If False then prints updates every 100 documents.

         Returns:
            (pd.DataFrame): Dataframe with new columns for the dataset. 
        """
        df, device = df_device_tuple

        # Load model and tokenizer to device
        self._load_model(device)
        self._load_tokenizer()

        # Reset device to model first layer if distributed with "auto"
        if device == "auto":
            first_layer_name = list(self.model.hf_device_map.keys())[0]
            device = self.model.hf_device_map[first_layer_name]

        # Get the indicators associated with each token
        label_tokens = [self.tokenizer.encode(label) for label in self.labels]
        token_indicators = [self.tokenizer.decode(tokens[1]) for tokens in label_tokens]
         
        new_cols = defaultdict(list)
        new_cols["prompt"] = df.apply(lambda row: prompt_func(**row), axis=1).tolist()

        for prompt in new_cols["prompt"]:
            if CoT:
                logprobs, cot_resp = self._label_one_CoT(device, prompt, 
                                                         top_logprobs, 
                                                         max_response_len)
                new_cols["CoT"].append(cot_resp)

            else:
                logprobs = self._label_one(device, prompt, top_logprobs)

            new_cols["logits"].append(logprobs)
            new_cols["LLM_raw_label"].append(logprobs[0][0])

            label_logprobs = get_logprobs(logprobs, token_indicators)
            label = self.labels[label_logprobs.argmax()]
            new_cols["LLM_top_logit"].append(label)
            new_cols["logits_clean"].append(label_logprobs)

            n_complete = len(new_cols["logits"])
            if (n_complete % 100 == 0):
                print(f"Completed {n_complete}/{len(new_cols['prompt'])} " \
                      f"observations on {device}.")

        # Assign new columns to dataframe
        df["LLM_raw_label"] = new_cols["LLM_raw_label"]
        df["LLM_top_logit"] = new_cols["LLM_top_logit"]
        df["logits"] = new_cols["logits"]
        df["prompt"] = new_cols["prompt"]
        df["logits_clean"] = new_cols["logits_clean"]

        if CoT:
            df["CoT"] = new_cols["CoT"]

        return df

    def _load_model(self, device):
        """
        Private class method to load the model and set self.model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                          token = self.hf_token, 
                                                          device_map = device)

    def _load_tokenizer(self):
        """
        Private class method to load the model tokenizer and set self.tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       token = self.hf_token)
        
    def _label_one(self, device, prompt, n_logits):
        """
        Private method to label one observation using the LLM.

        Args:
            - prompt (str): The string to prompt the LLM with.
            - k (int): The number of top-token logits to keep.

        Returns:
            - top_logits (List[Tuple(str, float)]): A list of the top token logits.
            - cot_resp (str): The Chain-of-Thought response.
        """
        chat = [{"role": "user", "content": prompt}]
        tokenized_chat = self.tokenizer.apply_chat_template(chat,
                                                            return_dict = True,
                                                            return_tensors="pt")
        
        input_ids = tokenized_chat["input_ids"].to(device)
        attention_mask = tokenized_chat["attention_mask"].to(device)

        model_out = self.model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = 1,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self.tokenizer.eos_token_id)
    
        # Get the top n_logits out
        topk = torch.topk(model_out["logits"][0], k = n_logits)
        tokens = [self.tokenizer.decode(tid) for tid in topk.indices[0]]
        logprobs = torch.log(torch.nn.functional.softmax(topk.values[0], dim = 0))
        top_logprobs = [(t, lp) for t, lp in zip(tokens, logprobs.tolist())]

        return top_logprobs
    
    def _label_one_CoT(self, device, prompt_pair, n_logits, max_response_len):
        """
        Private method to label one observation using CoT prompting.
        """
        chat = [{"role": "user", "content": prompt_pair[0]}]
        tokenized_chat = self.tokenizer.apply_chat_template(chat,
                                                            return_dict = True,
                                                            return_tensors="pt")

        input_ids = tokenized_chat["input_ids"].to(device)
        attention_mask = tokenized_chat["attention_mask"].to(device)
        
        model_cot = self.model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = max_response_len,
                                        return_dict_in_generate = True,
                                        pad_token_id = self.tokenizer.eos_token_id)

        # Get the response tokens and decode it
        cot_resp_tokens = model_cot["sequences"][0][input_ids[0].shape[0]:-1]
        cot_resp = self.tokenizer.decode(cot_resp_tokens)

        # Make new chat with response and query the model with it
        cot_chat = [{"role": "user", "content": prompt_pair[0]},
                    {"role": "assistant", "content": cot_resp},
                    {"role": "user", "content": prompt_pair[1]}]

        tokenized_chat = self.tokenizer.apply_chat_template(cot_chat,
                                                        return_dict = True,
                                                        return_tensors="pt")

        input_ids = tokenized_chat["input_ids"].to(device)
        attention_mask = tokenized_chat["attention_mask"].to(device)

        model_out = self.model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = 1,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self.tokenizer.eos_token_id)

        topk = torch.topk(model_out["logits"][0], n_logits)
        tokens = self.tokenizer.convert_ids_to_tokens(topk.indices[0])
        logprobs = torch.log(torch.nn.functional.softmax(topk.values[0], dim = 0))
        top_logprobs = [(t, lp) for t, lp in zip(tokens, logprobs.tolist())]

        return top_logprobs, cot_resp
