"""Tokenizer classes. Based on https://github.com/thunlp/OpenAttack/tree/master/OpenAttack/text_process/tokenizer."""

import transformers

from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from typing import List, Tuple, Union


class Tokenizer:
    """
    Tokenizer is the base class of all tokenizers.
    """

    def tokenize(self, x : str, pos_tagging : bool = True) -> Union[ List[str], List[Tuple[str, str]] ]:
        """
        Args:
            x: A sentence.
            pos_tagging: Whether to return Pos Tagging results.
        Returns:
            A list of tokens if **pos_tagging** is `False`
            
            A list of (token, pos) tuples if **pos_tagging** is `True`
        
        POS tag must be one of the following tags: ``["noun", "verb", "adj", "adv", "other"]``
        """
        return self.do_tokenize(x, pos_tagging)
    
    def detokenize(self, x : Union[List[str], List[Tuple[str, str]]]) -> str:
        """
        Args:
            x: The result of :py:meth:`.Tokenizer.tokenize`, can be a list of tokens or tokens with POS tags.
        Returns:
            A sentence.
        """
        if not isinstance(x, list):
            raise TypeError("`x` must be a list of tokens")
        if len(x) == 0:
            return ""
        x = [ it[0] if isinstance(it, tuple) else it for it in x ]
        return self.do_detokenize(x)

    
    def do_tokenize(self, x, pos_tagging):
        raise NotImplementedError()
    
    def do_detokenize(self, x):
        raise NotImplementedError()


_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv"
}


class PunctTokenizer(Tokenizer):
    """
    Tokenizer based on nltk.word_tokenizer.
    :Language: english
    """

    def __init__(self) -> None:
        self.sent_tokenizer = sent_tokenize
        self.word_tokenizer = WordPunctTokenizer().tokenize
        self.pos_tagger = PerceptronTagger()
        
    def do_tokenize(self, x, pos_tagging=True):
        sentences = self.sent_tokenizer(x)
        tokens = []
        for sent in sentences:
            tokens.extend( self.word_tokenizer(sent) )

        if not pos_tagging:
            return tokens
        ret = []
        for word, pos in self.pos_tagger(tokens):
            if pos[:2] in _POS_MAPPING:
                mapped_pos = _POS_MAPPING[pos[:2]]
            else:
                mapped_pos = "other"
            ret.append( (word, mapped_pos) )
        return ret

    def do_detokenize(self, x):
        return " ".join(x)


class TransformersTokenizer(Tokenizer):
    """
    Pretrained Tokenizer from transformers.
    Usually returned by :py:class:`.TransformersClassifier` .
    
    """

    def __init__(self, tokenizer : transformers.PreTrainedTokenizerBase):
        self.__tokenizer = tokenizer

    def do_tokenize(self, x, pos_tagging):
        if pos_tagging:
            raise ValueError("`%s` does not support pos tagging" % self.__class__.__name__)
        return self.__tokenizer.tokenize(x)
    
    def do_detokenize(self, x):
        return self.__tokenizer.convert_tokens_to_string(x)
