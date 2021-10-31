from nltk.corpus import wordnet as wn
from .misc import is_one_word

class WordSwap():
    def __init__(self):
        pass

    def get_new_word(self):
        raise NotImplementedError()

class WordSwapWordNet(WordSwap):
    def __init__(self, lang="eng"):
        assert lang in wn.langs(), f"\"{lang}\" language not present in WordNet languages."
        self.lang = lang

    def get_new_word(self, word):
        synonyms = set()
        synsets = wn.synsets(word, lang=self.lang)
        for syn in synsets:
            for syn_word in syn.lemma_names(lang=self.lang):
                if (
                    (syn_word != word)
                    and ("_" not in syn_word)
                    and (is_one_word(syn_word))
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.add(syn_word)
        return list(synonyms)
