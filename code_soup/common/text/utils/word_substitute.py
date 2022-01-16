"""
Contains different word subsitution methods such as replacing a word in a
sentence with its synonyms.
Adapted from
https://github.com/thunlp/OpenAttack/blob/master/OpenAttack/attack_assist/substitute/word/base.py.
"""
from nltk.corpus import wordnet as nltk_wn
from typing import List, Optional, Tuple

from code_soup.common.text.utils.exceptions import UnknownPOSException, WordNotInDictionaryException

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

POS_LIST = ["adv", "adj", "noun", "verb", "other"]


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


class WordSubstitute(object):
    def __call__(self, word : str, pos : Optional[str] = None) -> List[Tuple[str, float]]:
        """
        In WordSubstitute, we return a list of words that are semantically
        similar to the input word.
        
        Args:
            word: A single word.
            pos: POS tag of input word. Must be one of the following:
                 ``["adv", "adj", "noun", "verb", "other", None]``

        Raises:
            WordNotInDictionaryException: input word not in the dictionary of substitute algorithm
            UnknownPOSException: invalid pos tagging

        Returns:
            A list of words and their distance to original word
            (distance is a number between 0 and 1, with smaller indicating more
             similarity).
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


class WordNetSubstitute(WordSubstitute):

    def __init__(self, k=50):
        """
        English word substitute based on WordNet. WordNet is used to find
        synonyms (same named entity as the original word). 
        See Section 3.2.1 of the PWWS paper to get a better idea of how this works.
        Args:
            k: Top-k results to return. If k is `None`, all results will be
               returned. Default: 50   
        """

        self.wn = nltk_wn
        self.k = k

    def substitute(self, word: str, pos: str):
        """
        Finds candidate substitutes for the input word.

        Args:
            word (str): Input word (obtained after tokenising the input text).
            pos (str): POS tag (part of speech) of the input word (noun, verb,
                       etc.).

        Raises:
            WordNotInDictionaryException: If the word does not have a POS tag
                                          from list
                                          ["adv", "adj", "noun", "verb"].

        Returns:
            synonyms ([str]): List of candidate replacements.
        """
        token = word.replace('_', ' ').split()[0]
        if pos == "other":
            raise WordNotInDictionaryException()
        pos_in_wordnet = {
            "adv": "r",
            "adj": "a",
            "verb": "v",
            "noun": "n"
        }[pos]

        # Find synonyms using WordNet which belong to the same named entity.
        # Example (wordnet_synonyms for word "new"):
        """
        [Lemma('new.a.01.new'), Lemma('fresh.s.04.fresh'), Lemma('fresh.s.04.new'),
        Lemma('fresh.s.04.novel'), Lemma('raw.s.12.raw'), Lemma('raw.s.12.new'),
        Lemma('new.s.04.new'), Lemma('new.s.04.unexampled'), Lemma('new.s.05.new'),
        Lemma('new.a.06.new'), Lemma('newfangled.s.01.newfangled'),
        Lemma('newfangled.s.01.new'), Lemma('new.s.08.New'),
        Lemma('modern.s.05.Modern'), Lemma('modern.s.05.New'),
        Lemma('new.s.10.new'), Lemma('new.s.10.young'), Lemma('new.s.11.new')]
        """

        wordnet_synonyms = []
        synsets = self.wn.synsets(word, pos=pos_in_wordnet)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        # Preprocess the synonyms. Example:
        # {'young', 'novel', 'unexampled', 'new', 'fresh', 'newfangled', 'modern',
        #  'raw'}
        synonyms = set()
        for wordnet_synonym in wordnet_synonyms:
            # Step 1: Obtain the base word from the lemma.
            # Step 2: For multi-word synonyms, we only consider the first word.
            # Step 3: Prefilter the synonyms, i.e., remove words like "be", "is",
            #         "are", "am", etc.
            preprocessed_synonym = wordnet_synonym.name().split("_")[0]
            if prefilter(token, preprocessed_synonym):
                synonyms.add(preprocessed_synonym.lower())

        synonyms = [(syn, 1) for syn in synonyms]

        if self.k is not None and self.k > len(synonyms):
            synonyms = synonyms[:self.k]
        return synonyms
