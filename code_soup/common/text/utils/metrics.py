from typing import List
from code_soup.common.text.utils.tokenizer import Tokenizer

import torch

class AttackMetric(object):
    """
    Base class of all metrics.
    """

    def before_attack(self, input):
        return
    
    def after_attack(self, input, adversarial_sample):
        return

class Levenshtein(AttackMetric):

    def __init__(self, tokenizer : Tokenizer) -> None:
        """
        Args:
            tokenizer: A tokenizer that will be used in this metric. Must be an instance of :py:class:`.Tokenizer`
        """
        self.tokenizer = tokenizer
        self.name = "Levenshtein Edit Distance"
        
    def calc_score(self, a : List[str], b : List[str]) -> int:
        """
        Args:
            a: The first list.
            b: The second list.
        Returns:
            Levenshtein edit distance between two sentences.
            
        Both parameters can be str or list, str for char-level edit distance while list for token-level edit distance.
        """
        la = len(a)
        lb = len(b)
        f = torch.zeros(la + 1, lb + 1, dtype=torch.long)
        for i in range(la + 1):
            for j in range(lb + 1):
                if i == 0:
                    f[i][j] = j
                elif j == 0:
                    f[i][j] = i
                elif a[i - 1] == b[j - 1]:
                    f[i][j] = f[i - 1][j - 1]
                else:
                    f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
        return f[la][lb].item()

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score( self.tokenizer.tokenize(input["x"], pos_tagging=False), self.tokenizer.tokenize(adversarial_sample, pos_tagging=False) )