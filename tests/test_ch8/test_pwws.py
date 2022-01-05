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

def_tokenizer = PunctTokenizer()

path = "gchhablani/bert-base-cased-finetuned-sst2"

# define the attack
attacker = PWWSAttacker()

# define the victim model (classifier)
tokenizer = transformers.AutoTokenizer.from_pretrained(path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    path, num_labels=2, output_hidden_states=False)
victim = transformers_classifier.TransformersClassifier(model, tokenizer,
    model.bert.embeddings.word_embeddings)

# load the dataset
dataset = (datasets.load_dataset("sst", split="train[:2]").
    map(function=dataset_mapping))

# define the metric(s) which are to be computed between the original sample
# and the adversarial sample
metrics = [Levenshtein(def_tokenizer)]

result_iterator = attack_process(attacker, victim, dataset, metrics)


class TestPWWSAttacker(unittest.TestCase):
    """
    pwws.PWWSAttacker() test case
    """
    def setUp(cls):
        pass

    def test_output(self):
        res = next(result_iterator)

        x_orig = res["data"]["x"]
        x_adv = res["result"]

        probs = victim.get_prob([x_orig, x_adv])
        y_orig_prob = probs[0]
        y_adv_prob = probs[1]

        preds = victim.get_pred([x_orig, x_adv])                        
        y_orig_preds = int(preds[0])
        y_adv_preds = int(preds[1])
