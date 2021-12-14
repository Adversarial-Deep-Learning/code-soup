import random
import unittest

from parameterized import parameterized_class

from code_soup.common.text.utils import word_substitute
from code_soup.common.text.utils.exceptions import UnknownPOSException
from code_soup.misc import seed

seed(42)

@parameterized_class(
    ("word", "pos", "expected_result"),
    [("compute", "verb", [('calculate', 1), ('cipher', 1), ('figure', 1),
                          ('cypher', 1), ('work', 1), ('reckon', 1)]),
     ("bottle", "noun", [('bottleful', 1), ('feeding', 1), ('nursing', 1)])])
class TestWordNetSubstituteParameterized(unittest.TestCase):
    """
    WordNetSubstitute.substitute() Parameterized TestCase
    Args: ("word", "pos", "expected_result")
    """

    def setUp(self):
        self.wordnet_substitute = word_substitute.WordNetSubstitute()

    def test_output(self):
        self.assertEqual(
            sorted(self.wordnet_substitute.substitute(self.word, self.pos)),
            sorted(self.expected_result))

   
@parameterized_class(
    ("word", "pos", "expected_result"),
    [("compute", "verb", [('calculate', 1), ('cipher', 1), ('figure', 1),
                          ('cypher', 1), ('work', 1), ('reckon', 1)]),
     ("chair", None, [('hot', 1), ('electric', 1), ('death', 1), ('chairwoman', 1),
                      ('professorship', 1), ('chairman', 1), ('chairperson', 1),
                      ('president', 1)])])
class TestWordNetSubstituteCallParameterized(unittest.TestCase):
    """
    WordNetSubstitute() Parameterized TestCase
    Args: ("word", "pos", "expected_result")
    """

    def setUp(self):
        self.wordnet_substitute = word_substitute.WordNetSubstitute()

    def test_output(self):
        self.assertEqual(
            sorted(self.wordnet_substitute(self.word, self.pos)),
            sorted(self.expected_result))


class TestWordNetSubstituteCallException(unittest.TestCase):
    """
    WordNetSubstitute() TestCase for UnknownPOSException
    """

    def setUp(self):
        self.wordnet_substitute = word_substitute.WordNetSubstitute()

    def test_output(self):
        self.assertRaises(UnknownPOSException,
                          self.wordnet_substitute,"dummy", "none")