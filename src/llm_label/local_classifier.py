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

    Attributes:
        - model_name (str): The name of the model/tokenizer.
        - labels (List[str]): The target labels for each class.
        - hf_token (str): The hugging face token.
        - device (torch.device | str): The device to load and run the model on.
        - model (CausalLM): The huggingface model.
        - tokenizer (Tokenizer): The hugging face tokenizer.
        - label_token_ids (List[int]): A list of the token IDs associated with
            each model.
    """

    def __init__(self, model_name, labels, hf_token = None):
        """
        Initializes the LocalClassifer

        Args:
            - model_name (str): See Attributes
            - labels (List[str]): A list or iterable of the target class labels.
            - hf_token (str): The string for the huggingface token if required.
        """
        # Defined class attributes
        self.model_name = model_name
        self.hf_token = hf_token
        self.labels = list(labels)

        self.tokenizer = None
        self.model = None
        self.label_token_ids = None

    def label_df_on_device(self, df_device_tuple, prompt_func, CoT = False, 
                           max_response_len = 200):
        """
        Public method to label a df on a device.

        Args:
            - df_device_tuple tuple(pd.DataFrame:torch.device): 
                A tuple that contains the dataset to label and the device to
                to store the model for.
            - prompt_func (Callable()): Function that takes an observation and 
                target and creates a prompt (see prompting.py).
            - CoT (bool): Boolean indicator to perform CoT prompting.
            - max_response_len (int): The maximum number of tokens to return for
                CoT response if CoT is True.
            - progbar (bool): Boolean indicator to include progress bar
                for labeling. If False then prints updates every 100 documents.

         Returns:
            (pd.DataFrame): Dataframe with new columns for the dataset. 
        """
        df, device = df_device_tuple
        df = df.copy()

        # Load model and tokenizer to device
        self._load_model(device)

        # Reset device to model first layer if distributed with "auto"
        if device == "auto":
            first_layer_name = list(self.model.hf_device_map.keys())[0]
            device = self.model.hf_device_map[first_layer_name]
         
        new_cols = defaultdict(list)
        new_cols["prompt"] = df.apply(lambda row: prompt_func(**row), axis=1).tolist()

        for prompt in new_cols["prompt"]:
            if CoT:
                label_logprobs, raw_label, cot_resp = self._label_one_CoT(device, 
                                                                        prompt,
                                                                        max_response_len)
                new_cols["CoT"].append(cot_resp)

            else:
                label_logprobs, raw_label = self._label_one(device, prompt)

            new_cols["LLM_raw_label"].append(raw_label)
            new_cols["LLM_top_logit"].append(self.labels[label_logprobs.argmax()])
            new_cols["logits_clean"].append(label_logprobs.tolist())

            n_complete = len(new_cols["logits_clean"])
            if (n_complete % 100 == 0):
                print(f"Completed {n_complete}/{len(new_cols['prompt'])} " \
                      f"observations on {device}.")

        for key in new_cols.keys():
            df[key] = new_cols[key]

        return df

    def _load_model(self, device):
        """
        Loads the model, tokenzier and creates label_token_ids

        Args:
            - device (torch.device): The cuda device to load the model on. 
                Can be int, torch.device, or "auto" string!
                
        Returns: None. 
            Sets class attributes model, tokenzier, and label_token_ids
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                           token = self.hf_token,
                                                           device_map = device
                                                          )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Get the single token indicator associated with each label
            self.label_token_ids = []
            for label in self.labels:
                token_ids = self.tokenizer.encode(label, add_special_tokens=False)
                self.label_token_ids.append(token_ids[0])

        except ValueError as err:
            print(f"[!] Error loading model! {err}.")
            self.model = None
            self.tokenizer = None

    def _label_one(self, device, example):
        """
        Private method to label one example using the LLM.

        Args:
            - device (torch.device): The cuda device to load the model on. 
                Can be int, torch.device, or "auto" string!
            - example (str): The single example to prompt the LLM with.

        Returns:
            - label_logprobs (tensor): The tenosr of logprobs associated with
                each label's token ID.
            - raw_label (str): The raw text output from the model.
        """
        tokenized_example = self._tokenize_example(example)
        
        input_ids = tokenized_example["input_ids"].to(device)
        attention_mask = tokenized_example["attention_mask"].to(device)

        model_out = self.model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = 1,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self.tokenizer.eos_token_id)
        
        raw_label = self._get_text_response(model_out, input_ids[0].shape[0])
        logprobs = torch.log(torch.nn.functional.softmax(model_out.logits[0][0], dim=0))
        label_logprobs = logprobs[self.label_token_ids]
        
        return label_logprobs, raw_label
    
    def _label_one_CoT(self, device, prompt_pair, max_response_len):
        """
        Private method to label one observation using CoT prompting.

        Args:
            - device (torch.device): The cuda device to load the model on. 
                Can be int, torch.device, or "auto" string!
            - prompt_pair (Tuple[str]): The tuple of strings to create the CoT
                prompt with.

        Returns:
            - label_logprobs (tensor): The tenosr of logprobs associated with
                each label's token ID.
            - raw_label (str): The raw text output from the model.
            - cot_resp (str): The raw string from the model used fot CoT.
        """
        tokenized_example = self._tokenize_example(prompt_pair[0])
        
        input_ids = tokenized_example["input_ids"].to(device)
        attention_mask = tokenized_example["attention_mask"].to(device)
        
        model_out = self._model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = max_response_len,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self.tokenizer.eos_token_id)

        cot_resp = self._get_text_response(model_out, input_ids[0].shape[0])

        tokenzied_example = self._tokenize_example(prompt_pair, 
                                                   cot_response = cot_resp)

        input_ids = tokenized_example["input_ids"].to(device)
        attention_mask = tokenized_example["attention_mask"].to(device)

        model_out = self._model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = 1,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self.tokenizer.eos_token_id)
        
        raw_label = self._get_text_response(model_out, input_ids[0].shape[0])
        logprobs = torch.log(torch.nn.functional.softmax(model_out.logits[0][0], dim=0))
        label_logprobs = logprobs[self.label_token_ids]
        
        return label_logprobs, raw_label, cot_resp

    def _tokenize_example(self, example, cot_response = None):
        """ A helper function to tokenize a given example based on the model
            class passed.

        Args:
            - example (str): The example to tokenize.
            - cot_response (str): If provided a CoT response then the function
                                 will automatically assume the example is a 
                                 tuple and attempt to create a chat.

        Returns:
            - (dict): Dictionary with tokenized example and attention mask.
        """
        if cot_response is None:
                message = [{"role": "user", "content": example}]

        else: 
            message = [{"role": "user", "content": example[0]},
                        {"role": "assistant", "content": cot_response},
                        {"role": "user", "content": example[1]}]

        return self.tokenizer.apply_chat_template(message,
                                                  return_dict = True,
                                                  add_generation_prompt = True,
                                                  return_tensors = "pt")

    def _get_text_response(self, model_out, input_len):
        """ A helper function to return the model's response in plain-text.

        Args:
            - model_out : The response from the model.generate() call.
            - input_len : The number of tokens used to prompt the model.

        Returns:
            (str): The text response from the model.
        """
        model_out_tokens = model_out["sequences"][0][input_len:]
        return self.tokenizer.decode(model_out_tokens, 
                                     skip_special_tokens = True)


