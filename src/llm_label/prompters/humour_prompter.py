""" ideology_prompter.py

A prompter for the ideology datasets.

"""

from prompter import Prompter

class HumourPrompter(Prompter):

    """ Prompter for ideology detection.
    
    """

    def __init__(self, labels, column_map):
        super().__init__(labels)
        self.column_map = column_map


    def simple2(self, **kwargs):
        """ Returns simple prompt.
        
        """
        text = kwargs[self.column_map["text"]]

        prompt = f'{text} \n'
        prompt += "Is the above joke humorous to most people? You must pick between 'True' or 'False'. You cannot use any other words than 'True' or 'False'."

        return prompt

    def simple(self, **kwargs):
        """ Returns simple prompt.
        
        """
        text = kwargs[self.column_map["text"]]

        prompt = f'{text} \n'
        prompt += "Is the above joke humorous to most people? You must pick between 'True' or 'False'. You cannot use any other words than 'True' or 'False'."

        return prompt
        
if __name__ == "__main__":
    print("PRINTING EXAMPLE PROMPTS")

    labels = ["True", "False"]
    column_map = {"text": "joke"}
    prompter = HumourPrompter(labels, column_map)

    
    print(prompter.simple(**{"joke": "THIS IS AN EXAMPLE JOKE"}))