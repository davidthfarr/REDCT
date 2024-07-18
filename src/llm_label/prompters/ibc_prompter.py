""" ideology_prompter.py

A prompter for the ideology datasets.

"""

from .prompter import Prompter

class IBCPrompter(Prompter):
    """ Prompter for ideology detection.
    
    """
    
    def __init__(self, labels, column_map):
        super().__init__(labels)
        self.column_map = column_map

    def simple(self, **kwargs):
        """ Returns simple prompt.
        
        """
        text = kwargs[self.column_map["text"]]

        prompt = f'Statement: {text} \n'
        prompt += "Which of the following leanings would a political scientist " \
                  "say that the above statement has? \n"
        prompt += self._make_multiple_choice_str()
        prompt += self._make_constraint_str()

        return prompt
    
    def CoT(self):
        raise NotImplementedError
