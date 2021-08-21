import abc
import random


class ImportantWordExtractor(metaclass=abc.ABCMeta):
    """
    An abstract class used to represent the Keyword Extractor.SubClass should implement the extract method.

    Methods
    -------
    extract(self, words: list, **kwargs)
        returns the index of important words in the list.
    """

    @abc.abstractmethod
    def extract(self, words: list, **kwargs):  # pragma: no cover
        """Extracts Important Word and returns the Index."""
        raise NotImplementedError

    def empty_error_msg(self):
        return "given list of words is empty or invalid"

    def word_less_than_k(self):
        return "number of words should be greater than top_k"


class RandomImportantWordExtractor(ImportantWordExtractor):
    """
    A class used to apply random word extractor.

    Methods
    -------
    extract(self, words: list, top_k = 1, **kwargs)
        - extracts top_k random words and returns their index.

    Parameters
    --------
    words - List of words.
    top_k - Number of words to be extracted
    """

    def extract(self, words: list, top_k=1, **kwargs):

        assert len(words) > 0, self.empty_error_msg()
        assert len(words) >= top_k, self.word_less_than_k()

        return random.sample(range(len(words)), top_k)
