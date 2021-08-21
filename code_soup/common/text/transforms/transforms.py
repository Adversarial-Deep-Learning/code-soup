import abc

from code_soup.common.text.extractor import basic
from code_soup.common.text.transforms import perturbations

__all__ = ["AddChar", "ShuffleChar", "DeleteChar", "TypoChar", "VisuallySimilarChar"]

extractor_list = [
    "RandomWordExtractor"
]  # Add more extractors here after they have been implemented.


class Transforms(object):
    """
    Parent class for all transforms.

    Methods
    -------
    apply(self, text: str, extractor: code_soup.common.text.extractor.basic.ImportantWordExtractor(
    perturb_type: CharacterPerturbations, **kwargs)
            -Applies the respective transform (depending on the subclass which calls it)
    extractor_not_valid_message(self)
            -Display Message if the extractor string is not a valid extractor
    """

    def extractor_not_valid_message(self):
        return "Extractor chosen invalid. Please choose from " + str(extractor_list)

    def apply(self, text, extractor, perturb_type, **kwargs):
        words = text.split(" ")
        indices = extractor.extract(words)

        for index in indices:
            words[index] = perturb_type.apply(words[index], **kwargs)

        return " ".join(word for word in words)


class Compose(object):
    """
    Composes several transforms together.

    Args:
            transforms: list
                    -list of transforms to execute.

    Example:
            tfms=transforms.Compose([
                    transforms.AddChar("RandomWordExtractor", True),
                    transforms.VisuallySimilarChar("RandomWordExtractor")
                                                            ])
            transformed=tfms("This is fascinating!")
            print(transformed)
            This i̅s̅ faschinating!
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        for t in self.transforms:
            text = t(text)

        return text

    def __repr__(self):
        format_string = self.__class__.__name__ + "("

        for t in self.transforms:
            format_string += "\n"
            format_string += "	{0}".format(t)

        format_string += "\n"
        return format_string


class AddChar(Transforms):
    """
    To the extracted words, add space/character randomly.

    Args:
    extractor: str (default: "RandomWordCharacter")
            -One of ["RandomWordExtractor"]
    char_perturb: boolean (default: False)
            -If True, add space in word randomly.
            -If False, add character in word randomly.
    ignore: boolean (default: True)
            -If True, ignore assertion errors (recommended).
            -If False, do not ignore assertion errors.

    Example:
    tfms=transforms.AddChar(extractor="RandomWordExtractor", char_perturb=False)
    transformed=tfms("This is fascinating!")
    print(transformed)
    This i s fascinating!
    """

    def __init__(
        self, extractor="RandomWordExtractor", char_perturb=False, ignore=True
    ):

        assert extractor in ["RandomWordExtractor"], self.extractor_not_valid_message()

        if extractor == "RandomWordExtractor":
            self.extractor = basic.RandomImportantWordExtractor()

        self.char_perturb = char_perturb
        self.space_char_perturb = perturbations.InsertSpaceCharacterPerturbations()
        self.ignore = ignore

    def __call__(self, text):
        kwargs = {"char_perturb": self.char_perturb, "ignore": self.ignore}
        return self.apply(text, self.extractor, self.space_char_perturb, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ShuffleChar(Transforms):
    """
        Randomly shuffle characters in the chosen words.

        Args:
        extractor: str (default: "RandomWordExtractor")
                -One of ["RandomWordExtractor"]
        mid: boolean (default: False)
                -if True, shuffles the characters of a word at random, barring the initial and last character
    -if False, swaps any two characters of a word at random, barring the initial and last character
        ignore: boolean (default: True)
                -If True, ignore assertion errors (recommended).
                -If False, do not ignore assertion errors.

        Example:
        tfms=transforms.ShuffleChar(extractor="RandomWordExtractor", mid=False)
        transformed=tfms("This is fascinating!")
        print(transformed)
        This is fascinatign!
    """

    def __init__(self, extractor="RandomWordExtractor", mid=False, ignore=True):

        assert extractor in ["RandomWordExtractor"], self.extractor_not_valid_message()

        if extractor == "RandomWordExtractor":
            self.extractor = basic.RandomImportantWordExtractor()

        self.shuffle_char_perturb = perturbations.ShuffleCharacterPerturbations()
        self.mid = mid
        self.ignore = ignore

    def __call__(self, text):
        kwargs = {"mid": self.mid, "ignore": self.ignore}
        return self.apply(text, self.extractor, self.shuffle_char_perturb, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class DeleteChar(Transforms):
    """
    Delete characters randomly from the chosen words.

    Args:
    extractor: str (default: "RandomWordExtractor")
            -One of ["RandomWordExtractor"]
    ignore: boolean (default: True)
            -If True, ignore assertion errors (recommended).
            -If False, do not ignore assertion errors.

    Example:
    tfms=transforms.DeleteChar(extractor="RandomWordExtractor")
    transformed=tfms("This is fascinating!")
    print(transformed)
    This is fascinting!
    """

    def __init__(self, extractor="RandomWordExtractor", ignore=True):

        assert extractor in ["RandomWordExtractor"], self.extractor_not_valid_message()

        if extractor == "RandomWordExtractor":
            self.extractor = basic.RandomImportantWordExtractor()

        self.delete_char_perturb = perturbations.DeleteCharacterPerturbations()
        self.ignore = ignore

    def __call__(self, text):
        kwargs = {"ignore": self.ignore}
        return self.apply(text, self.extractor, self.delete_char_perturb, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class TypoChar(Transforms):
    """
    Introduces typographical errors in the chosen words

    Args:
    extractor: str (default: "RandomWordExtractor")
            -One of ["RandomWordExtractor"]
    probability: float in range [0,1] (default: 0.1)
            -probability*100 percent characters in the word will become typos.
    ignore: boolean (default: True)
            -If True, ignore assertion errors (recommended).
            -If False, do not ignore assertion errors.

    Example:
    tfms=transforms.TypoChar(extractor="RandomWordExtractor", probability=0.1)
    transformed=tfms("This is fascinating!")
    print(transformed)
    This us fascinating!
    """

    def __init__(self, extractor="RandomWordExtractor", probability=0.1, ignore=True):

        assert extractor in ["RandomWordExtractor"], self.extractor_not_valid_message()

        if extractor == "RandomWordExtractor":
            self.extractor = basic.RandomImportantWordExtractor()

        self.typo_char_perturb = perturbations.TypoCharacterPerturbations()
        self.probability = probability
        self.ignore = ignore

    def __call__(self, text):
        kwargs = {"probability": self.probability, "ignore": self.ignore}
        return self.apply(text, self.extractor, self.typo_char_perturb, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class VisuallySimilarChar(Transforms):
    """
    Implement VisuallySimilarCharacterPerturbations on the extracted words

    Args:
    extractor: str (default: "RandomWordExtractor")
            -One of ["RandomWordExtractor"]
    seed: int (default: None)
            -seed for random
    ignore: boolean (default: True)
            -If True, ignore assertion errors (recommended).
            -If False, do not ignore assertion errors.

    Example:
    tfms=transforms.VisuallySimilarChar("RandomWordCharacter", None)
    transformed=tfms("This is fascinating!")
    print(transformed)
    T̕h̒i̕s̒ is fascinating!
    """

    def __init__(self, extractor="RandomWordExtractor", seed=None, ignore=True):

        assert extractor in ["RandomWordExtractor"], self.extractor_not_valid_message()

        if extractor == "RandomWordExtractor":
            self.extractor = basic.RandomImportantWordExtractor()

        self.visually_similar_char_perturb = (
            perturbations.VisuallySimilarCharacterPerturbations("unicode", "homoglyph")
        )
        self.ignore = ignore
        self.seed = seed

    def __call__(self, text):
        kwargs = {"seed": self.seed, "ignore": self.ignore}
        return self.apply(
            text, self.extractor, self.visually_similar_char_perturb, **kwargs
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"
