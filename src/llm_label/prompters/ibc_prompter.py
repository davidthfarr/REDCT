""" ideology_prompter.py

A prompter for the ideology datasets.

"""

from prompter import Prompter

class IBCPrompter(Prompter):

    """ Prompter for ideology detection.
    
    """

    def __init__(self, labels, column_map):
        super().__init__(labels)
        self.column_map = column_map


    def simple2(self, **kwargs):
        """ Returns simple prompt.
        
        """
        text = kwargs[self.column_map["text"]]

        prompt = "There are two leading political ideologies: Liberal and Conservative. " \
           "Liberals favor progressive social policies and goverment-funded social " \
           "welfare programs, while conservatives tend to favor traditional values " \
           "and limited government intervention. Which of the following leanings " \
           "would a political scientist say that the following statement has? \n"
        prompt += f'statement: "{text}" \n'
        prompt += self._make_constraint_str()
        prompt += "Respond 'Neutral' if there is no strong polotical leaning."

        return prompt

    def simple(self, **kwargs):
        """ Returns simple prompt.
        
        """
        text = kwargs[self.column_map["text"]]

        #prompt = "You are a political science expert. Please evaluate the following text and determine if it is more conservative or liberal.  \n"
        prompt = f'Statement: {text} \n'
        prompt += "Which of the following leanings would a political scientist say that the above statement has? \n A: Conservative \n B: Neutral \n C: Liberal \n"
        prompt += self._make_constraint_str()

        return prompt

if __name__ == "__main__":
    print("PRINTING EXAMPLE PROMPTS")

    labels = ["Conservative", "Neutral", "Liberal"]
    column_map = {"text": "sentence"}
    prompter = IBCPrompter(labels, column_map)

    
    print(prompter.simple2(**{"sentence": "THIS IS AN EXAMPLE HEADLINE"}))