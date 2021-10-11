import random
import unittest

from parameterized import parameterized_class

from code_soup.common.text import perturbations

WHITE_SPACE_EXAMPLE = "is wrong"


@parameterized_class(
    ("word", "expected_result"), [("Bob", "Bb"), ("Hey there", "Hey there"), ("H", "H")]
)
class TestPerturbDeleteParameterized(unittest.TestCase):
    """Perturb Delete Parameterized TestCase
    Args: ("word", "expected_result")
    """

    def setUp(self):
        random.seed(42)
        self.delete_perturbations = perturbations.DeleteCharacterPerturbations()

    def test_output(self):
        self.assertEqual(
            self.delete_perturbations.apply(self.word), self.expected_result
        )


class TestPerturbDeleteUnparameterized(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.delete_perturbations = perturbations.DeleteCharacterPerturbations()

    def test_perturb_delete_with_character_size_less_than_three(self):
        with self.assertRaises(AssertionError):
            self.delete_perturbations.apply("To", ignore=False)

    def test_perturb_delete_with_whitespace(self):
        with self.assertRaises(AssertionError):
            self.delete_perturbations.apply(WHITE_SPACE_EXAMPLE, ignore=False)


@parameterized_class(
    ("word", "expected_result"),
    [("Bob", "B ob"), ("Hey there", "Hey there"), ("H", "H")],
)
class TestPerturbInsertSpaceParameterized(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.space_perturb = perturbations.InsertSpaceCharacterPerturbations()

    def test_output(self):
        self.assertEqual(self.space_perturb.apply(self.word), self.expected_result)


class TestPerturbInsertSpaceUnparameterized(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.space_perturb = perturbations.InsertSpaceCharacterPerturbations()

    def test_perturb_insert_space_with_whitespace(self):
        with self.assertRaises(AssertionError):
            self.space_perturb.apply(WHITE_SPACE_EXAMPLE, ignore=False)

    def test_perturb_insert_space_with_character_size_less_than_two(self):
        with self.assertRaises(AssertionError):
            self.space_perturb.apply("H", ignore=False)


@parameterized_class(
    ("word", "expected_result"),
    [("hello", "hellzo"), ("Hey there", "Hey there"), ("H", "H")],
)
class TestPerturbInsertCharacter(unittest.TestCase):
    def setUp(self):
        random.seed(30)
        self.char_perturb = perturbations.InsertSpaceCharacterPerturbations()

    def test_output(self):
        self.assertEqual(
            self.char_perturb.apply(self.word, char_perturb=True), self.expected_result
        )


@parameterized_class(("word", "expected_result"), [("THAT", "TAHT")])
class TestPerturbShuffleSwapTwo(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.shuffle_perturbations = perturbations.ShuffleCharacterPerturbations()

    def test_output(self):
        self.assertEqual(
            self.shuffle_perturbations.apply(self.word, mid=False), self.expected_result
        )


@parameterized_class(
    ("word", "expected_result"), [("Adversarial", "Aiavrsedarl"), ("dog", "dog")]
)
class TestPerturbShuffleMiddle(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.shuffle_perturbations = perturbations.ShuffleCharacterPerturbations()

    def test_output(self):
        self.assertEqual(
            self.shuffle_perturbations.apply(self.word), self.expected_result
        )


class TestPerturbShuffleUnparameterized(unittest.TestCase):
    def setUp(self):
        self.shuffle_perturbations = perturbations.ShuffleCharacterPerturbations()

    def test_perturb_shuffle_with_character_size_less_than_four(self):
        with self.assertRaises(AssertionError):
            self.shuffle_perturbations.apply("Ton", ignore=False)

    def test_perturb_shuffle_with_whitespace(self):
        with self.assertRaises(AssertionError):
            self.shuffle_perturbations.apply(WHITE_SPACE_EXAMPLE, ignore=False)


@parameterized_class(
    ("word", "expected_result"), [("Noise", "Noixe"), ("Hi there", "Hi there")]
)
class TestPerturbTypo(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.typo_perturbations = perturbations.TypoCharacterPerturbations()

    def test_output(self):
        self.assertEqual(self.typo_perturbations.apply(self.word), self.expected_result)


class TestPerturbTypoWithWhitespace(unittest.TestCase):
    def setUp(self):
        self.typo_perturbations = perturbations.TypoCharacterPerturbations()

    def test_perturb_shuffle_with_whitespace(self):
        with self.assertRaises(AssertionError):
            self.typo_perturbations.apply(
                WHITE_SPACE_EXAMPLE, probability=0.1, ignore=False
            )


@parameterized_class(
    ("word", "expected_result"),
    [("adversarial", "âd́v̕e̕r̕s̐a̕r̂i̅a̒ĺ"), ("Hi there", "Hi there")],
)
class TestPerturbUnicode(unittest.TestCase):
    def setUp(self):
        self.viz = perturbations.VisuallySimilarCharacterPerturbations(
            "unicode", "homoglyph"
        )

    def test_output(self):
        self.assertEqual(self.viz.apply(self.word, 0), self.expected_result)


@parameterized_class(
    ("word", "expected_result"),
    [("adversarial", "𝓪𝓭ꮩ𝑒𝓇ｓ𝖺rꙇa1"), ("Hi there", "Hi there")],
)
class TestPerturbHomoglyph(unittest.TestCase):
    def setUp(self):
        self.viz = perturbations.VisuallySimilarCharacterPerturbations(
            "unicode", "homoglyph"
        )

    def test_output(self):
        self.assertEqual(self.viz.apply(self.word, 1), self.expected_result)
