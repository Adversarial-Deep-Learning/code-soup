import random
import unittest

from parameterized import parameterized_class

from code_soup.common.text.utils import metrics, tokenizer
from code_soup.misc import seed


@parameterized_class(
    ("input", "adversarial_sample", "expected_output"),
    [({"x": "compute"}, "comp te", 2), ({"x": "bottle"}, "abossme", 1)],
)
class TestLevenshteinParameterized(unittest.TestCase):
    """
    Levenshtein.after_attack Parameterized test case
    Args: ("input", "adversarial_sample", "expected_output")
    """

    def setUp(self):
        self.levenshtein = metrics.Levenshtein(tokenizer.PunctTokenizer())

    def test_output(self):
        self.assertEqual(
            self.levenshtein.after_attack(self.input, self.adversarial_sample),
            self.expected_output,
        )
