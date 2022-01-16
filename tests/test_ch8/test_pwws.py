import datasets
import transformers
import unittest

from parameterized import parameterized_class


from code_soup.ch8.pwws import PWWSAttacker
from code_soup.common.text.datasets.utils import dataset_mapping
from code_soup.common.text.models import transformers_classifier
from code_soup.common.text.utils.attack_helpers import attack_process
from code_soup.common.text.utils.metrics import Levenshtein
from code_soup.common.text.utils.tokenizer import PunctTokenizer
from code_soup.misc import seed

seed(42)


class TestPWWSAttacker(unittest.TestCase):
    """
    pwws.PWWSAttacker() test cases
    """
    @classmethod
    def setUpClass(cls) -> None:
        def_tokenizer = PunctTokenizer()

        path = "gchhablani/bert-base-cased-finetuned-sst2"

        # define the attack
        cls.attacker = PWWSAttacker()

        # define the victim model (classifier)
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=2, output_hidden_states=False)
        cls.victim = transformers_classifier.TransformersClassifier(model, tokenizer,
            model.bert.embeddings.word_embeddings)

        # load the dataset
        cls.dataset = (datasets.load_dataset("sst", split="train[:2]").
            map(function=dataset_mapping))

    def test_output(cls):
        for sample in cls.dataset:
            cls.attacker(cls.victim, sample)
