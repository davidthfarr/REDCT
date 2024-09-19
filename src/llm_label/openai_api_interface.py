""" @openai_api_interface.py

A python module to interface with the Open AI API and get valid labels from
the provided model.

"""

import openai
import tiktoken
from tqdm import tqdm

class OpenAPI:
    """
    A simple class to interact with the OpenAI API for question-answer prompts.
    """

    def __init__(self, api_key, model_name = "gpt-3.5-turbo"):
        """
        Initialize the class with the Open AI API with the following args:
            api_key (str): Your secret API key.
            model_name (str): The model you want access to.

        Other model attributes:
            self.responses (List[response]): List of raw json responses.
        """
        self.client = openai.OpenAI(api_key = api_key)
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.responses = []

    def request_prompt(self, prompt_list, logprobs = True, top_logprobs = 20, 
                       logit_bias = None, verbose = True):
        """
        A function to send requests to the open AI chat API. Please refer to the
        Open AI documentation https://platform.openai.com/docs/api-reference/chat.
        Args:
            - prompt_list (List[str]): List of prompts.
            - logprobs (bool): Indicator to return logprobs or not.
            - top_logprobs (int): Number of top predictions to return.
            - logit_bias (Dict): The logit bias for predicting.
        Returns:
            self.responses.
        """
        for prompt in tqdm(prompt_list, disable = (not verbose)):
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages = [{"role": "user", "content": prompt}],
                logprobs = logprobs,
                top_logprobs = top_logprobs,
                logit_bias = logit_bias,
                temperature = 0
            )
            self.responses.append(resp)
    
        return self.responses
    
    def parse_responses(self, responses_raw = None):
        """
        Function that parses all responses in the response list or given list.
        Args: responses_raw (optional: List[responses])
        Returns: (Dict[str:List]) Parsed responses in dictionary.
        """
        if responses_raw is None:
            responses = self.responses

        else:
            responses = responses_raw

        parsed = {'Label': [], "Logits": []}

        for resp in responses:
            parsed['Label'].append(resp.choices[0].message.content)
            raw_logits = resp.choices[0].logprobs.content[0].top_logprobs
            parsed["Logits"].append([(r.token, r.logprob) for r in raw_logits])
        
        return parsed

    def get_token_IDs(self, tokens):
        """
        Helper function to get the token IDs for the particular model you are 
        using. Particularly useful to get the token IDs for the logit_bias.
        Args:
            tokens (List[str]): List of strings to get token IDs for.
        Returns: 
            token_IDs (List[Int]): List of token IDs associated with each token.
        """
        return [self.tokenizer.encode(token) for token in tokens]

    def decode_token_IDs(self, token_ids):
        """
        Helper function to get the tokens back from the token IDs.

        Args:
            token_ids (List[int]): List of integer token IDs
        
        Returns:
            tokens (List[str]): String of tokens.
        """
        return [self.tokenizer.decode([token_id]) for token_id in token_ids]

class OpenAPI_CoT(OpenAPI):
    """
    A simple class to interact with the OpenAI API for Chain of Thought prompts.
    """

    def __init__(self, api_key, model_name = "gpt-3.5-turbo"):
        """
        Initialize the class with the Open AI API with the following args:
            api_key (str): Your secret API key.
            model_name (str): The model you want access to.

        Other model attributes:
            self.responses (List[List(responses)]): List of response pairs from
            the API.
        """
        super().__init__(api_key, model_name)

    def request_prompt(self, prompt_list, logprobs = True, top_logprobs = 20, 
                       logit_bias = None, verbose = True):
        """
        A function to send requests to the open AI chat API. Please refer to the
        Open AI documentation https://platform.openai.com/docs/api-reference/chat.
        Args:
            - prompt_list (List[Tuple[str]]): List of CoT prompts.
            - logprobs (bool): Indicator to return logprobs or not.
            - top_logprobs (int): Number of top predictions to return.
            - logit_bias (Dict): The logit bias for predicting.
        Returns:
            self.responses.
        """
        for prompt1, prompt2 in tqdm(prompt_list, disable = (not verbose)):
            both_resp = []

            resp1 = self.client.chat.completions.create(
                model=self.model_name,
                messages = [{"role": "user", "content": f"{prompt1}"}],
            )

            both_resp.append(resp1)
            resp1_text = resp1.choices[0].message.content

            resp2 = self.client.chat.completions.create(
                model=self.model_name,
                messages = [{"role": "user", "content": prompt1},
                            {"role": "assistant", "content": resp1_text},
                            {"role": "user", "content": prompt2}],
                logprobs = logprobs,
                top_logprobs = top_logprobs,
                logit_bias = logit_bias,
                temperature = 0
            )

            both_resp.append(resp2)
            self.responses.append(both_resp)
    
        return self.responses
    
    def parse_responses(self, responses_raw = None):
        """
        Function that parses all responses in the response list or given list.
        Args: responses_raw (optional: List[responses])
        Returns: (Dict[str:List]) Parsed responses in dictionary.
        """
        if responses_raw is None:
            responses = self.responses

        else:
            responses = responses_raw

        parsed = {'CoT': [], 'Label': [], "Logits": []}

        for resp1, resp2 in responses:
            parsed['CoT'].append(resp1.choices[0].message.content)
            parsed['Label'].append(resp2.choices[0].message.content)
            raw_logits = resp2.choices[0].logprobs.content[0].top_logprobs
            parsed["Logits"].append([(r.token, r.logprob) for r in raw_logits])
        
        return parsed
    