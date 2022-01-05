import unittest

from parameterized import parameterized_class
from transformers import BertTokenizer

from code_soup.common.text.utils import tokenizer


@parameterized_class(
    ("x", "expected_result"),
    [("xlnet is better than bert . but bert has less parameters .",
        [('xlnet', 'noun'), ('is', 'verb'), ('better', 'adj'), ('than', 'other'),
         ('bert', 'noun'), ('.', 'other'), ('but', 'other'), ('bert', 'noun'),
         ('has', 'verb'), ('less', 'adj'), ('parameters', 'noun'), ('.', 'other')]),
     ("reformers are efficient transformers . longformers can handle long texts .",
        [('reformers', 'noun'), ('are', 'verb'), ('efficient', 'adj'),
         ('transformers', 'noun'), ('.', 'other'), ('longformers', 'noun'),
         ('can', 'other'), ('handle', 'verb'), ('long', 'adj'), ('texts', 'noun'),
         ('.', 'other')])])
class TestPunctTokenizerTokenizeWPosParameterized(unittest.TestCase):
    """
    PunctTokenizer.tokenize() Parameterized TestCase
    Args: ("x", "expected_result")
    """

    def setUp(self):
        self.tok = tokenizer.PunctTokenizer()

    def test_output(self):
        self.assertEqual(self.tok.tokenize(self.x), self.expected_result)


@parameterized_class(
    ("x", "expected_result"),
    [("xlnet is better than bert . but bert has less parameters .",
        ['xlnet', 'is', 'better', 'than', 'bert', '.', 'but', 'bert', 'has', 'less', 'parameters', '.']),
     ("reformers are efficient transformers . longformers can handle long texts .",
        ['reformers', 'are', 'efficient', 'transformers', '.', 'longformers', 'can', 'handle', 'long', 'texts', '.'])])
class TestPunctTokenizerTokenizeWoPosParameterized(unittest.TestCase):
    """
    PunctTokenizer.tokenize() Parameterized TestCase
    Args: ("x", "expected_result")
    """

    def setUp(self):
        self.tok = tokenizer.PunctTokenizer()

    def test_output(self):
        self.assertEqual(self.tok.tokenize(self.x, False), self.expected_result)


@parameterized_class(
    ("x", "expected_result"),
    [(['xlnet', 'is', 'better', 'than', 'bert', '.', 'but', 'bert', 'has', 'less', 'parameters', '.'], "xlnet is better than bert . but bert has less parameters ."),
     (['reformers', 'are', 'efficient', 'transformers', '.', 'longformers', 'can', 'handle', 'long', 'texts', '.'], "reformers are efficient transformers . longformers can handle long texts ."),
     ([], "")])
class TestPunctTokenizerDetokenizeParameterized(unittest.TestCase):
    """
    PunctTokenizer.tokenize() Parameterized TestCase
    Args: ("x", "expected_result")
    """

    def setUp(self):
        self.tok = tokenizer.PunctTokenizer()

    def test_output(self):
        self.assertEqual(self.tok.detokenize(self.x), self.expected_result)


@parameterized_class(
    ("x", "expected_result"),
    [("short sentence .",
        ['short', 'sentence', '.']),
     ("another sentence, slightly longer .",
        ['another', 'sentence', ',', 'slightly', 'longer', '.'])])
class TestTransformersTokenizerTokenizeParameterized(unittest.TestCase):
    """
    TransformersTokenizer.tokenize() Parameterized TestCase
    Args: ("x", "expected_result")
    """

    def setUp(self):
        self.tok = tokenizer.TransformersTokenizer(BertTokenizer.from_pretrained("bert-base-uncased"))

    def test_output(self):
        self.assertEqual(self.tok.tokenize(self.x, False), self.expected_result)

@parameterized_class(
    ("x", "expected_result"),
    [(['short', 'sentence', '.'], "short sentence ."),
     (['another', 'sentence', ',', 'slightly', 'longer', '.'], "another sentence, slightly longer .")])
class TestTransformersTokenizerDetokenizeParameterized(unittest.TestCase):
    """
    TransformersTokenizer.detokenize() Parameterized TestCase
    Args: ("x", "expected_result")
    """

    def setUp(self):
        self.tok = tokenizer.TransformersTokenizer(BertTokenizer.from_pretrained("bert-base-uncased"))

    def test_output(self):
        self.assertEqual(self.tok.detokenize(self.x), self.expected_result)
