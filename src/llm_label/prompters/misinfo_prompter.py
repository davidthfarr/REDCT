""" misinfo_prompter.py

Module to build prompts for zero-shot misinformation classification.
A lot of this is based on Cruickshank and Ng "Prompting and Fine-Tuning 
Open-Sourced Language Models for Stance Classification," and Ziems et al. 
"Can Large Language Models Transform Computational Social Science?"

We follow their naming convention for prompting techniques.

"""

from prompter import Prompter 

class MisinfoPrompter(Prompter):
    """ Prompter for misinformation detection.
    
    """

    def __init__(self, labels, column_map):
        super().__init__(labels)
        self.column_map = column_map


    def simple(self, **kwargs):
        """ Returns simple prompt.
        
        """
        text = kwargs[self.column_map["text"]]

        prompt = f'"{text}" \n'
        prompt += "Which of the following describes the above news headline? \n" 
        prompt += self._make_multiple_choice_str() + self._make_constraint_str()

        return prompt

if __name__ == "__main__":
    print("PRINTING EXAMPLE PROMPTS")

    labels = ["Misinformation", "Trustworthy"]
    column_map = {"text": "headline"}
    prompter = MisinfoPrompter(labels, column_map)

    
    print(prompter.simple(**{"headline": "THIS IS AN EXAMPLE HEADLINE"}))

    
        