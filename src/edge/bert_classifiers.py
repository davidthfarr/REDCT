""" @bert_models.py

Python module that contains the general class for a bert_models.

"""

import torch
from transformers import DistilBertModel

class BertBasedClassifier(torch.nn.Module):
    """
    Class that contains the structure for a transformer based classifier.
    """
    def __init__(self, base_pretrained_encoder, model_name, num_classes):
        """
        Initialize the model with the hugging face base_pretrained_encoder class
        from the transformers and define structure of each layer.

        - base_pretrained_encoder (class) is a class from the transformers
            library with valid from_pretrained() method.
        - model_name (str): The name of the model for the from pre-trained method.
        """
        self.base_pretrained_encoder = base_pretrained_encoder
        super(BertBasedClassifier, self).__init__()

        self.bert = self.base_pretrained_encoder.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.lin_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.out_layer = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoding = bert_out.pooler_output
        lin_out = self.lin_layer(encoding)
        activation = torch.nn.ReLU()(lin_out)
        drop = self.dropout(activation)
        output = self.out_layer(drop)
        return output

class DistilBertClassifier(BertBasedClassifier):
    """
    Class which creates a classifier from DistilBert. The same as the BertBasedClassifier
    but we have to change the forward method slightly.
    """
    def __init__(self, num_classes):
        """
        The class is initialized the same as the base classifier.
        """
        super(DistilBertClassifier, self).__init__(DistilBertModel, 
                                                   "distilbert-base-cased",
                                                  num_classes)
    
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_out[0]
        encoding = hidden_state[:, 0]
        lin_out = self.lin_layer(encoding)
        activation = torch.nn.ReLU()(lin_out)
        drop = self.dropout(activation)
        output = self.out_layer(drop)
        return output
