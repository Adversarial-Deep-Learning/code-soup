"""PWWS Attack implementation. The code has been adapted from https://github.com/thunlp/OpenAttack/blob/master/OpenAttack/attackers/pwws/__init__.py."""

from typing import List, Optional
import numpy as np

from code.soup.common.text.utils.classification_model_wrappers import Classifier
from code_soup.common.text.utils.exceptions import WordNotInDictionaryException
from code_soup.common.text.utils.misc import ENGLISH_FILTER_WORDS
from code_soup.common.text.utils.tokenizer import Tokenizer, get_default_tokenizer
from code_soup.common.text.utils.word_substitute import WordNetSubstitute


class PWWSAttacker:

    def __init__(self,
            victim : Classifier,
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
        
    def attack(self, victim: Classifier, sentence : str, goal : ClassifierGoal):
        x_orig = sentence.lower()


        x_orig = self.tokenizer.tokenize(x_orig)
        poss =  list(map(lambda x: x[1], x_orig)) 
        x_orig =  list(map(lambda x: x[0], x_orig))

        S = self.get_saliency(victim, x_orig, goal) # (len(sent), )
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        w_star = [ self.get_wstar(victim, x_orig, i, poss[i], goal) for i in range(len(x_orig)) ]  # (len(sent), )
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
            if goal.check(curr_sent, pred):
                return curr_sent
        return None


    
    def get_saliency(self, clsf, sent, goal : ClassifierGoal):
        """
        Get saliency scores for every score. Simply put, saliency score of a word is the degree of change in the
        output probability of the classifier if the word is set to unknown (out of vocabulary). See Section 3.2.2
        in the paper for more details.

        Args:
            clsf (Classifier): A classifier that will be used to get the saliency scores.
            sent (list): List of tokens in a sentence.
            goal: A classifier goal that will be used to check if the sentence is a valid one.
        """
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1:]
            # Replace the word with unknown token
            x_i_hat = left + [self.token_unk] + right
            x_hat_raw.append(self.tokenizer.detokenize(x_i_hat))
        x_hat_raw.append(self.tokenizer.detokenize(sent))
        res = clsf.get_prob(x_hat_raw)[:, goal.target]
        if not goal.targeted:
            res = res[-1] - res[:-1]
        else:
            res = res[:-1] - res[-1]
        return res

    def get_wstar(self, clsf, sent, idx, pos, goal : ClassifierGoal):
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
        res = clsf.get_prob(sents)[:, goal.target]
        prob_orig = res[-1]
        res = res[:-1]
        if goal.targeted:
            return (rep_words[ res.argmax() ],  res.max() - prob_orig )
        else:
            return (rep_words[ res.argmin() ],  prob_orig - res.min() )
