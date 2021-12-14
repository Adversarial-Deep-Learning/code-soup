import numpy as np
import random
import torch
import unittest

from parameterized import parameterized_class
from transformers import BertForSequenceClassification, BertTokenizer

from code_soup.common.text.models.transformers_classifier import TransformersClassifier
from code_soup.misc import seed

seed(42)
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
embedding_layer = model.bert.embeddings.word_embeddings
device = torch.device("cpu")

@parameterized_class(
    ("input_", "expected_output"),
    [(["inception is an awesome movie ."], [1]),
     (["marvel is cliche .", "Fascinating movie, that !"],
       [0, 1])])
class TestTransformersClassifierGetPred(unittest.TestCase):
    """
    Parameterized test cases for the TransformersClassifier.get_pred() function
    from the common/text/models/transformers_classifier.py file.

    Args: ("x", "expected_output")
    """
    def setUp(self):
        self.clf = TransformersClassifier(model, tokenizer, embedding_layer, device)

    def test_output(self):
        self.assertEqual(list(self.clf.get_pred(self.input_)), self.expected_output)

@parameterized_class(
    ("input_", "expected_output"),
    [(["inception is an awesome movie ."], np.array([[0.01, 0.99]])),
     (["marvel is cliche .", "Fascinating movie, that !"],
       np.array([[0.997, 0.003], [0.032, 0.968]]))])
class TestTransformersClassifierGetProb(unittest.TestCase):
    """
    Parameterized test cases for the TransformersClassifier.get_prob() function
    from the common/text/models/transformers_classifier.py file.

    Args: ("x", "expected_output")
    """
    def setUp(self):
        self.clf = TransformersClassifier(model, tokenizer, embedding_layer, device)

    def test_output(self):
        self.assertIsNone(np.testing.assert_almost_equal(
            self.clf.get_prob(self.input_), self.expected_output, decimal=3))
