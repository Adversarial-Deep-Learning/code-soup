from nltk.corpus import wordnet as nltk_wn
from typing import List, Optional, Tuple

from code_soup.common.text.utils.exceptions import UnknownPOSException, WordNotInDictionaryException

import nltk
nltk.download('wordnet')

POS_LIST = ["adv", "adj", "noun", "verb", "other"]

class WordSubstitute(object):
    def __call__(self, word : str, pos : Optional[str] = None) -> List[Tuple[str, float]]:
        """
        In WordSubstitute, we return a list of words that are semantically similar to the input word.
        
        Args:
            word: A single word.
            pos: POS tag of input word. Must be one of the following: ``["adv", "adj", "noun", "verb", "other", None]``
        
        Returns:
            A list of words and their distance to original word (distance is a number between 0 and 1, with smaller indicating more similarity)
        Raises:
            WordNotInDictionaryException: input word not in the dictionary of substitute algorithm
            UnknownPOSException: invalid pos tagging
        """
        
        if pos is None:
            ret = {}
            for sub_pos in POS_LIST:
                try:
                    for word, sim in self.substitute(word, sub_pos):
                        if word not in ret:
                            ret[word] = sim
                        else:
                            ret[word] = max(ret[word], sim)
                except WordNotInDictionaryException:
                    continue
            list_ret = []
            for word, sim in ret.items():
                list_ret.append((word, sim))
            if len(list_ret) == 0:
                raise WordNotInDictionaryException()
            return sorted( list_ret, key=lambda x: -x[1] )
        elif pos not in POS_LIST:
            raise UnknownPOSException("Invalid `pos` %s (expect %s)" % (pos, POS_LIST) )
        return self.substitute(word, pos)
    
    def substitute(self, word : str, pos : str) -> List[Tuple[str, float]]:
        raise NotImplementedError()


def prefilter(token, synonym):  # 预过滤（原词，一个候选词
    if (len(synonym.split()) > 2 or (  # the synonym produced is a phrase
            synonym == token) or (  # the pos of the token synonyms are different
            token == 'be') or (
            token == 'is') or (
            token == 'are') or (
            token == 'am')):  # token is be
        return False
    else:
        return True


class WordNetSubstitute(WordSubstitute):

    def __init__(self, k = None):
        """
        English word substitute based on wordnet.
        Args:
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
        
        :Data Requirements: :py:data:`.TProcess.NLTKWordNet`
        :Language: english
        
        """

        self.wn = nltk_wn
        self.k = k

    def substitute(self, word: str, pos: str):
        if pos == "other":
            raise WordNotInDictionaryException()
        pos_in_wordnet = {
            "adv": "r",
            "adj": "a",
            "verb": "v",
            "noun": "n"
        }[pos]

        wordnet_synonyms = []
        synsets = self.wn.synsets(word, pos=pos_in_wordnet)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())
        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = wordnet_synonym.name().replace('_', ' ').split()[0]
            synonyms.append(spacy_synonym)  # original word
        token = word.replace('_', ' ').split()[0]

        sss = []
        for synonym in synonyms:
            if prefilter(token, synonym):
                sss.append(synonym)
        synonyms = sss[:]

        synonyms_1 = []
        for synonym in synonyms:
            if synonym.lower() in synonyms_1:
                continue
            synonyms_1.append(synonym.lower())

        ret = []
        for syn in synonyms_1:
            ret.append((syn, 1))
        if self.k is not None:
            ret = ret[:self.k]
        return ret