import random

import pytest

from code_soup.common.text.extractor import basic


@pytest.mark.parametrize(
    "words, expected_result",
    [(["This", "is", "test"], [1]), (["Hey", "there"], [1])],
)
def test_random_extract(words, expected_result):
    random.seed(0)
    random_extractor = basic.RandomImportantWordExtractor()
    assert random_extractor.extract(words) == expected_result


@pytest.mark.parametrize(
    "words, expected_result", [(["This", "is", "a", "test"], [3, 1])]
)
def test_random_k_extract(words, expected_result):
    random.seed(0)
    random_extractor = basic.RandomImportantWordExtractor()
    assert random_extractor.extract(words, top_k=2) == expected_result


def test_random_empty_extract():
    random_extractor = basic.RandomImportantWordExtractor()
    with pytest.raises(AssertionError):
        random_extractor.extract([])


def test_words_less_than_k_extract():
    random_extractor = basic.RandomImportantWordExtractor()
    with pytest.raises(AssertionError):
        random_extractor.extract(["Hey", "There"], top_k=3)
