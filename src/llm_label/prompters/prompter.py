""" prompter.py

Module to help build prompts for zero-shot classification.

"""

import string

class Prompter:
    """ Basic prompt. 
    
    Must be inherented by all subclasses for specific problems and/or datasets.

    Attributes: 
        - num_classes (int): The number of classes to be predicted.
        - labels(List[str]): List of labels in display order.
    """

    def __init__(self, labels):
        """ Init prompt.

        Args:
            - labels(List[str]): List of labels in display order.
        """
        self.num_classes = len(labels)
        self.labels = list(labels)

    def _make_multiple_choice_str(self):
        """ Makes a multiple choice string from the label map

        Args: None
        Returns: (str): Formated multiple choice string.
        """
        out_str = ""
        for idx, label in enumerate(self.labels):
            out_str += f"{string.ascii_uppercase[idx]}) {label} \n"

        return out_str

    def _make_class_parenthesis(self):
        """ Gets all the classes and puts them in a string of parentheses.

        Args: None
        Returns: (str): Formated string of labels.
        """
        out_str = ""
        for idx, label in enumerate(self.labels):
            if idx == 0:
                out_str += f"({label}, "
                
            elif idx == (self.num_classes - 1):
                out_str += f"or {label})"

            else:
                out_str += f"{label}, "

        return out_str

    def _make_constraint_str(self):
        """ Makes a contraint string based on the label map
        
        Args: None
        Returns: (str): Formated constraint string.
        """
        out_str = "Only respond with one word " 

        for idx, label in enumerate(self.labels):
            if idx == (self.num_classes - 1):
                out_str += f" or '{label}'."

            else:
                out_str += f"'{label}', "
        
        return out_str
