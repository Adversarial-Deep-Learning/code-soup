"""
PWWS Attack implementation. The code has been adapted from
https://github.com/thunlp/OpenAttack/blob/master/OpenAttack/attackers/pwws/__init__.py.
"""


import sys

sys.path.append("./")

from typing import Any, Optional

import datasets
import numpy as np
import transformers

from code_soup.common.text.datasets.utils import dataset_mapping
from code_soup.common.text.models import classifier, transformers_classifier
from code_soup.common.text.utils.attack_helpers import *
from code_soup.common.text.utils.exceptions import WordNotInDictionaryException
from code_soup.common.text.utils.metrics import *
from code_soup.common.text.utils.misc import ENGLISH_FILTER_WORDS
from code_soup.common.text.utils.tokenizer import PunctTokenizer, Tokenizer
from code_soup.common.text.utils.word_substitute import WordNetSubstitute


def check(prediction, target, targeted):
    """
    A utility function to check if the attack was successful. If the attack is
    targeted, then the "predicted" class must be same as "target" class.
    Otherwise, the "predicted" class must be different from the "target" class.

    Args:
        prediction (int): Predicted class (as returned by the model).
        target (int): Has a dual meaning. If targeted = True, then target is the
                      class we want the model to predict (on the adversarial
                      sample). Otherwise, target is the class the model predicted
                      for the original sample.
        targeted (bool): Whether the attack is targeted or not. Targeted attack
                         here means that we want to obtain an adversarial sample
                         such that the model predicts the specified target class.

    Returns:
        (bool): Returns whether the attack was successful or not.
    """
    if targeted:
        return prediction == target
    else:
        return prediction != target


class PWWSAttacker:
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_unk: str = "<UNK>",
    ):
        """
        Generating Natural Language Adversarial Examples through Probability
        Weighted Word Saliency.
        Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019.

        `[pdf] <https://www.aclweb.org/anthology/P19-1103.pdf>`__
        `[code] <https://github.com/JHL-HUST/PWWS/>`__

        Args:
            tokenizer: A tokenizer that will be used during the attack procedure.
                       Must be an instance of Tokenizer
            token_unk: The token id or the token name for out-of-vocabulary
                       words in victim model. Default: <UNK>
        """
        # PWWS attack substitutes words using synonyms obtained from WordNet.
        # For a detailed description of the method, please refer to Section 3.2.1.
        # You can also refer to code_soup/ch8/common/text/utils/word_substitute.py.
        self.substitute = WordNetSubstitute()

        if tokenizer is None:
            tokenizer = PunctTokenizer()
        self.tokenizer = tokenizer

        self.token_unk = token_unk
        self.filter_words = set(ENGLISH_FILTER_WORDS)

    def __call__(self, victim: classifier.Classifier, input_: Any):
        """
        Generates the adversarial sample when the attacker object is called.

        Args:
            victim (classifier.Classifier): A classifier which is to be attacked.
            input_ (Any): A dictionary which contains the input data (text and
                          label). Example:
                          {'label': 0.625,
                           'x': 'Singer\\/composer Bryan Adams contributes a
                                 slew of songs .',
                            'y': 1
                          }

        Raises:
            RuntimeError: If the attack is not successful.

        Returns:
            adversarial_sample (str): Adversarial sample generated by PWWS.
        """
        # If the attack is targeted
        if "target" in input_:
            target = input_["target"]
            targeted = True
        # If the attack is not targeted, keep the target as the predicted label
        # of the original text; in untargeted attack, we will generate a sample
        # with predicted label different from the predicted label of the
        # original text.
        else:
            target = victim.get_pred([input_["x"]])[0]
            targeted = False

        # Generate the adversarial sample.
        adversarial_sample = self.attack(victim, input_["x"], target, targeted)

        if adversarial_sample is not None:
            # Obtain the predicted label of the adversarial sample.
            y_adv = victim.get_pred([adversarial_sample])[0]
            # Verify if the attack was successful. If not, raise an error.
            if not check(y_adv, target, targeted):
                raise RuntimeError(
                    "Check attacker result failed: "
                    "result ([%d] %s) expect (%s%d)"
                    % (y_adv, adversarial_sample, "" if targeted else "not ", target)
                )
        return adversarial_sample

    def attack(
        self, victim: classifier.Classifier, sentence: str, target=0, targeted=True
    ):
        """
        Given an input sample, generate the adversarial text.

        Args:
            victim (classifier.Classifier): A classifier which is to be attacked.
            sentence (str): Input text.
            target (int): Has a dual meaning. If targeted = True, then target is
                          the class we want the model to predict (on the
                          adversarial sample). Otherwise, target is the class
                          the model predicted for the original sample.  Defaults
                          to 0.
            targeted (bool): Whether the attack is targeted or not. Targeted
                            attack here means that we want to obtain an adversarial
                            sample such that the model predicts the specified
                            target class. Defaults to True.

        Returns:
            (str): Adversarial sample generated by PWWS.
        """
        # Example of x_orig: "inception is an awesome movie ."
        x_orig = sentence.lower()
        # Words: ['inception', 'is', 'an', 'awesome', 'movie', '.']
        # POS Tags: ['noun', 'verb', 'other', 'adj', 'noun', 'other']

        # Obtain words and their respective POS tags.
        x_orig_pos = self.tokenizer.tokenize(x_orig)
        x_orig, poss = list(map(list, zip(*x_orig_pos)))

        # Get the saliency score for every word in the input text. Example:
        # [1.19209290e-06, 4.29153442e-06, 1.41859055e-05, 5.17034531e-03,
        # 7.03334808e-06, 4.76837158e-07]
        S = self.get_saliency(victim, x_orig, target, targeted)
        # Normalise the saliency scores. Example:
        # [0.16652223, 0.16652276, 0.16652441, 0.16738525, 0.1665232, 0.16652212]
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        # Obtain the best replacement word for every word in the input text.
        # Example:
        # [('origination', -2.3841858e-07), ('is', 0), ('an', 0),
        #  ('awful', 0.9997573), ('pic', 1.180172e-05), ('.', 0)]
        w_star = [
            self.get_wstar(victim, x_orig, i, poss[i], target, targeted)
            for i in range(len(x_orig))
        ]
        # Compute "H" score for every word. It is simply the product of the w_star
        # score and the saliency scores. See Eqn (7) in the paper. Example:
        # [(0, 'origination', -3.9701995e-08), (1, 'is', 0.0),
        #  (2, 'an', 0.0), (3, 'awful', 0.16734463),
        #  (4, 'pic', 1.9652603e-06), (5, '.', 0.0)]
        H = [
            (idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1])
            for idx in range(len(x_orig))
        ]

        # Sort the words in the input text by their "H" score (descending order).
        H = sorted(H, key=lambda x: -x[2])
        ret_sent = x_orig.copy()
        for i in range(len(H)):
            idx, wd, _ = H[i]
            if ret_sent[idx] in self.filter_words:
                continue
            ret_sent[idx] = wd

            curr_sent = self.tokenizer.detokenize(ret_sent)
            pred = victim.get_pred([curr_sent])[0]
            # Verify if the attack was successful.
            if check(pred, target, targeted):
                return curr_sent
        return None

    def get_saliency(
        self, clsf: classifier.Classifier, sent: List[str], target=0, targeted=True
    ):
        """
        Get saliency scores for every score. Simply put, saliency score of a
        word is the degree of change in the output probability of the classifier
        if the word is set to unknown (out of vocabulary). See Section 3.2.2
        in the paper for more details.

        Args:
            clsf (Classifier): A classifier that will be used to get the
                               saliency scores.
            sent (list): List of tokens in a sentence.
            target (int): Has a dual meaning. If targeted = True, then target is
                          the class we want the model to predict (on the
                          adversarial sample). Otherwise, target is the class
                          the model predicted for the original sample.  Defaults
                          to 0.
            targeted (bool): Whether the attack is targeted or not. Targeted
                            attack here means that we want to obtain an adversarial
                            sample such that the model predicts the specified
                            target class. Defaults to True.
        """
        # Replace words with <UNK> one by one. Compute probability for every such
        # sample.
        # Example: sent = ["inception", "is", "an", "awesome", "movie", "."]
        # A few samples generated: ['<UNK>', 'is', 'an', 'awesome', 'movie', '.'],
        # ['inception', '<UNK>', 'an', 'awesome', 'movie', '.'], etc.
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1 :]
            # Replace the word with unknown token
            x_i_hat = left + [self.token_unk] + right
            x_hat_raw.append(self.tokenizer.detokenize(x_i_hat))
        # Concatenate the original text as well; we want to compute the probability
        # for the original sample too (because we want the difference in probs)
        # between generated samples and original sample).
        x_hat_raw.append(self.tokenizer.detokenize(sent))

        # Compute the probabilities. Example:
        # [0.9999354, 0.9999323, 0.9999224, 0.99476624, 0.99992955, 0.9999361,
        #  0.9999366]. Clearly, the 4th element of the list differs the most
        # from the last element (probability of the original sample). The 4th
        # element is the probability of ["inception", "is", "an", "<UNK>", "movie", "."].
        # This proves that the word "awesome" plays a major role in determining
        # the classification output.
        res = clsf.get_prob(x_hat_raw)[:, target]
        if not targeted:
            res = res[-1] - res[:-1]
        else:
            res = res[:-1] - res[-1]
        return res

    def get_wstar(
        self,
        clsf: classifier.Classifier,
        sent: List[str],
        idx: int,
        pos: str,
        target=0,
        targeted=True,
    ):
        """
        Given a word in a sentence, find the replacment word (from a list of
        candidate replacements) that maximises the difference in probabilities
        between the original sample and the generated sample (generated sample
        is the sample with the word replaced by the candidate word). This score
        is given as delta(P) in the paper. See Section 3.2.1 for more details.

        Args:
            clsf (classifier.Classifier): A classifier which is to be attacked.
            sent ([str]): Input text.
            idx (int): Index of word in sentence.
            pos (str): POS Tag.
            target (int): Has a dual meaning. If targeted = True, then target is
                          the class we want the model to predict (on the
                          adversarial sample). Otherwise, target is the class
                          the model predicted for the original sample.  Defaults
                          to 0.
            targeted (bool): Whether the attack is targeted or not. Targeted
                            attack here means that we want to obtain an adversarial
                            sample such that the model predicts the specified
                            target class. Defaults to True.

        Returns:
            ((str, float)): Best replacement word (w_star) and its score (delta(P)
                            in the paper).
        """
        # Example: sent = ["inception", "is", "an", "awesome", movie, "."]
        # idx = 3, word = "awesome", pos = "adj"
        # Its replacement words are: ['awing', 'amazing', 'awful', 'awe-inspiring']
        word = sent[idx]
        try:
            # Obtain replacement words.
            rep_words = list(map(lambda x: x[0], self.substitute(word, pos)))
        except WordNotInDictionaryException:
            rep_words = []
        # Remove the word itself from the list of replacement words.
        rep_words = list(filter(lambda x: x != word, rep_words))
        # If there are no replacement words, return the original word with score 0.
        if len(rep_words) == 0:
            return (word, 0)

        sents = []
        for rw in rep_words:
            # Step 1: Replace word with candidate word.
            new_sent = sent[:idx] + [rw] + sent[idx + 1 :]
            sents.append(self.tokenizer.detokenize(new_sent))
        # Append the original sentence as well, we want to compute the difference
        # in probabilities between original sample and generated samples.
        sents.append(self.tokenizer.detokenize(sent))
        # Get the probabilities. Example:
        # Word: awesome
        # rep_words: ['awe-inspiring', 'awful', 'awing', 'amazing']
        # [5.1087904e-01, 9.9993670e-01, 9.9991834e-01, 1.7930799e-04, 9.9993658e-01]
        res = clsf.get_prob(sents)[:, target]
        prob_orig = res[-1]
        res = res[:-1]
        # Find the best replacement word, i.e., w_star. We maximise delta(P) here.
        # Clearly, the best replacement word is the 4th word, i.e., awing.
        if targeted:
            return (rep_words[res.argmax()], res.max() - prob_orig)
        else:
            return (rep_words[res.argmin()], prob_orig - res.min())


# Example
def main():
    def_tokenizer = PunctTokenizer()

    path = "gchhablani/bert-base-cased-finetuned-sst2"

    # define the attack
    attacker = PWWSAttacker()

    # define the victim model (classifier)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2, output_hidden_states=False
    )
    victim = transformers_classifier.TransformersClassifier(
        model, tokenizer, model.bert.embeddings.word_embeddings
    )

    # load the dataset
    dataset = datasets.load_dataset("sst", split="train[:10]").map(
        function=dataset_mapping
    )

    # define the metric(s) which are to be computed between the original sample
    # and the adversarial sample
    metrics = [Levenshtein(def_tokenizer)]

    result_iterator = attack_process(attacker, victim, dataset, metrics)

    total_inst = 0
    success_inst = 0

    for i, res in enumerate(result_iterator):
        try:
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
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()