"""PWWS Attack implementation. The code has been adapted from https://github.com/thunlp/OpenAttack/blob/master/OpenAttack/attackers/pwws/__init__.py."""
import datasets

from typing import Any, Optional
import numpy as np

from code_soup.common.text.utils.metrics import *
from code_soup.common.text.utils.attack_helpers import *
from code_soup.common.text.datasets.utils import dataset_mapping
from code_soup.common.text.models import classifier, transformers_classifier
from code_soup.common.text.utils.exceptions import WordNotInDictionaryException
from code_soup.common.text.utils.misc import ENGLISH_FILTER_WORDS
from code_soup.common.text.utils.tokenizer import Tokenizer, PunctTokenizer
from code_soup.common.text.utils.word_substitute import WordNetSubstitute
from code_soup.common.text.utils.visualizer import visualizer

import sys
import transformers

def check(prediction, target, targeted):
    if targeted:
        return prediction == target
    else:
        return prediction != target


class PWWSAttacker:

    def __init__(self,
            tokenizer : Optional[Tokenizer] = None,
            token_unk : str = "<UNK>",
        ):
        """
        Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency. Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019.
        `[pdf] <https://www.aclweb.org/anthology/P19-1103.pdf>`__
        `[code] <https://github.com/JHL-HUST/PWWS/>`__
        Args:
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            filter_words: A list of words that will be preserved in the attack procedure.
        :Classifier Capacity:
            * get_pred
            * get_prob
        """

        self.substitute = WordNetSubstitute()

        if tokenizer is None:
            tokenizer = PunctTokenizer()
        self.tokenizer = tokenizer

        self.token_unk = token_unk
        self.filter_words = set(ENGLISH_FILTER_WORDS)

    def __call__(self, victim: classifier.Classifier, input_: Any):

        if "target" in input_:
            target = input_["target"]
            targeted = True
        else:
            target = victim.get_pred([ input_["x"] ])[0]
            targeted = False
        
        adversarial_sample = self.attack(victim, input_["x"], target, targeted)

        if adversarial_sample is not None:
            y_adv = victim.get_pred([ adversarial_sample ])[0]
            if not check(y_adv, target, targeted):
                raise RuntimeError("Check attacker result failed: result ([%d] %s) expect (%s%d)" % ( y_adv, adversarial_sample, "" if targeted else "not ", target))
        return adversarial_sample
        
    def attack(self, victim: classifier.Classifier, sentence : str, target=0, targeted=True):
        x_orig = sentence.lower()


        x_orig = self.tokenizer.tokenize(x_orig)
        poss =  list(map(lambda x: x[1], x_orig)) 
        x_orig =  list(map(lambda x: x[0], x_orig))

        S = self.get_saliency(victim, x_orig, target, targeted) # (len(sent), )
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        w_star = [ self.get_wstar(victim, x_orig, i, poss[i], target, targeted) for i in range(len(x_orig)) ]  # (len(sent), )
        H = [ (idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1]) for idx in range(len(x_orig)) ]

        H = sorted(H, key=lambda x:-x[2])
        ret_sent = x_orig.copy()
        for i in range(len(H)):
            idx, wd, _ = H[i]
            if ret_sent[idx] in self.filter_words:
                continue
            ret_sent[idx] = wd
            
            curr_sent = self.tokenizer.detokenize(ret_sent)
            pred = victim.get_pred([curr_sent])[0]
            if check(pred, target, targeted):
                return curr_sent
        return None


    def get_saliency(self, clsf, sent, target=0, targeted=True):
        """
        Get saliency scores for every score. Simply put, saliency score of a word is the degree of change in the
        output probability of the classifier if the word is set to unknown (out of vocabulary). See Section 3.2.2
        in the paper for more details.

        Args:
            clsf (Classifier): A classifier that will be used to get the saliency scores.
            sent (list): List of tokens in a sentence.
        """
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1:]
            # Replace the word with unknown token
            x_i_hat = left + [self.token_unk] + right
            x_hat_raw.append(self.tokenizer.detokenize(x_i_hat))
        x_hat_raw.append(self.tokenizer.detokenize(sent))
        res = clsf.get_prob(x_hat_raw)[:, target]
        if not targeted:
            res = res[-1] - res[:-1]
        else:
            res = res[:-1] - res[-1]
        return res

    def get_wstar(self, clsf, sent, idx, pos, target=0, targeted=True):
        word = sent[idx]
        try:
            rep_words = list(map(lambda x:x[0], self.substitute(word, pos)))
        except WordNotInDictionaryException:
            rep_words = []
        rep_words = list(filter(lambda x: x != word, rep_words))
        if len(rep_words) == 0:
            return ( word, 0 )
        sents = []
        for rw in rep_words:
            new_sent = sent[:idx] + [rw] + sent[idx + 1:]
            sents.append(self.tokenizer.detokenize(new_sent))
        sents.append(self.tokenizer.detokenize(sent))
        res = clsf.get_prob(sents)[:, target]
        prob_orig = res[-1]
        res = res[:-1]
        if targeted:
            return (rep_words[ res.argmax() ],  res.max() - prob_orig )
        else:
            return (rep_words[ res.argmin() ],  prob_orig - res.min() )


def main():
    def_tokenizer = PunctTokenizer()
    path = "BERT.SST" # change path
    attacker = PWWSAttacker()

    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, output_hidden_states=False)
    victim = transformers_classifier.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)
    metrics = [Levenshtein(def_tokenizer)]

    result_iterator = attack_process(attacker, victim, dataset, metrics)

    total_result = {}
    total_result_cnt = {}
    total_inst = 0
    success_inst = 0

    for i, res in enumerate(result_iterator):
        total_inst += 1
        success_inst += int(res["success"])

        x_orig = res["data"]["x"]
        x_adv = res["result"]

        probs = victim.get_prob([x_orig, x_adv])
        y_orig_prob = probs[0]
        y_adv_prob = probs[1]

        preds = victim.get_pred([x_orig, x_adv])                        
        y_orig_preds = int(preds[0])
        y_adv_preds = int(preds[1])

        print("======================================================")
        print(f"{i}th sample")
        print("Original: ")
        print(f"TEXT: {x_orig}")
        print(f"Probabilities: {y_orig_prob}")
        print(f"Predictions: {y_orig_preds}")
        
        print("Adversarial: ")
        print(f"TEXT: {x_adv}")
        print(f"Probabilities: {y_adv_prob}")
        print(f"Predictions: {y_adv_preds}")
        
        print("\nMetrics: ")
        print(res["metrics"])
        print("======================================================")
