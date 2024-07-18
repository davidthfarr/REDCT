""" ideology_prompter.py

A prompter for the ideology datasets.

"""

from .prompter import Prompter

class HumourPrompter(Prompter):
    """ Prompter for humour detection.
    
    """

    def __init__(self, labels, column_map):
        super().__init__(labels)
        self.column_map = column_map

    def simple(self, **kwargs):
        """ Returns simple prompt.
        
        """
        text = kwargs[self.column_map["text"]]

        prompt = f"{text} \n"
        prompt += "Is the above joke humorous to most people? You must pick " \
                  "between 'True' or 'False'. "
        prompt += self._make_constraint_str()

        return prompt
    
    def CoT(self):
        raise NotImplementedError
