"""PWWS Attack implementation. The code has been adapted from https://github.com/thunlp/OpenAttack/blob/master/OpenAttack/attackers/pwws/__init__.py."""

from typing import List, Optional
import numpy as np

from code_soup.common.text.utils.exceptions import WordNotInDictionaryException
from code_soup.common.text.utils.misc import ENGLISH_FILTER_WORDS
from code_soup.common.text.utils.tokenizer import Tokenizer, get_default_tokenizer
from code_soup.common.text.utils.word_substitute import WordNetSubstitute


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
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        self.token_unk = token_unk
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        
    def attack(self, victim: Classifier, sentence : str, target=0, targeted=True):
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



import numpy as np
import datasets
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# configure access interface of the customized victim model by extending OpenAttack.Classifier.
class MyClassifier:
    def __init__(self):
        # nltk.sentiment.vader.SentimentIntensityAnalyzer is a traditional sentiment classification model.
        nltk.download('vader_lexicon')
        self.model = SentimentIntensityAnalyzer()
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        ret = []
        for sent in input_:
            # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance
            res = self.model.polarity_scores(sent)

            # we use 𝑠𝑜𝑐𝑟𝑒_𝑝𝑜𝑠 / (𝑠𝑐𝑜𝑟𝑒_𝑛𝑒𝑔 + 𝑠𝑐𝑜𝑟𝑒_𝑝𝑜𝑠) to represent the probability of positive sentiment
            # Adding 10^−6 is a trick to avoid dividing by zero.
            prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)

            ret.append(np.array([1 - prob, prob]))
        
        # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
        return np.array(ret)

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
def main():
    # load some examples of SST-2 for evaluation
    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
    # choose the costomized classifier as the victim model
    victim = MyClassifier()
    # choose PWWS as the attacker and initialize it with default parameters
    attacker = PWWSAttacker()
    # prepare for attacking
    attack_eval = OpenAttack.AttackEval(attacker, victim)
    # launch attacks and print attack results 
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()