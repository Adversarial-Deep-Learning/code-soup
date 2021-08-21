import math
import random

import pytest

from code_soup.common.text.metrics import char_metrics


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [("Word", "Wordy", 1), ("Word", "Wrod", 2), ("H", "H", 0)],
)
def test_levenshtein(text1, text2, expected_result):
    levenshtein_distance = char_metrics.Levenshtein()
    assert levenshtein_distance.calculate(text1, text2) == expected_result


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [("Word", "Wordy", 0.1111111111111111), ("Word", "Wrod", 0.25), ("H", "H", 0)],
)
def test_levenshtein_sum(text1, text2, expected_result):
    levenshtein_distance = char_metrics.Levenshtein()
    err = levenshtein_distance.calculate(text1, text2, "sum") - expected_result
    assert -1e-5 < err < 1e-5


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [("Word", "Wordy", 0.2), ("Word", "Wrod", 0.5), ("H", "H", 0)],
)
def test_levenshtein_lcs(text1, text2, expected_result):
    levenshtein_distance = char_metrics.Levenshtein()
    err = levenshtein_distance.calculate(text1, text2, "lcs") - expected_result
    assert -1e-5 < err < 1e-5


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [("a cat", "an abct", 3), ("a cat", "a tc", 2)],
)
def test_damerau_levenshtein(text1, text2, expected_result):
    damerau_levenshtein_distance = char_metrics.DamerauLevenshtein()
    assert damerau_levenshtein_distance.calculate(text1, text2) == expected_result


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [("a cat", "an abct", 0.42857142857142855), ("a cat", "a tc", 0.4)],
)
def test_damerau_levenshtein_lcs(text1, text2, expected_result):
    damerau_levenshtein_distance = char_metrics.DamerauLevenshtein()
    err = damerau_levenshtein_distance.calculate(text1, text2, "lcs") - expected_result
    assert -1e-5 < err < 1e-5


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [("a cat", "an abct", 0.25), ("a cat", "a tc", 0.2222222222222222)],
)
def test_damerau_levenshtein_sum(text1, text2, expected_result):
    damerau_levenshtein_distance = char_metrics.DamerauLevenshtein()
    err = damerau_levenshtein_distance.calculate(text1, text2, "sum") - expected_result
    assert -1e-5 < err < 1e-5


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [
        ("Word", "Wordy", 1.4142135623730951),
        ("Word was", "Word is that", 1.7320508075688772),
        ("H", "H", 0),
    ],
)
def test_euclid(text1, text2, expected_result):
    euclidean_distance = char_metrics.Euclidean()
    err = euclidean_distance.calculate(text1, text2) - expected_result
    assert -1e-5 < err < 1e-5


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [
        ("Word", "Wordy", 1),
        ("Word was", "Word is that", 0.8660254037844386),
        ("H", "H", 0),
    ],
)
def test_euclid_norm(text1, text2, expected_result):
    euclidean_distance = char_metrics.Euclidean()
    err = euclidean_distance.calculate(text1, text2, norm=True) - expected_result
    assert -1e-5 < err < 1e-5


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [
        ("Word", "Wordy", 0.19999999999999996),
        ("Word", "Wrod", 0),
        ("Word was", "Word is that", 0.36363636363636365),
    ],
)
def test_jaccard(text1, text2, expected_result):
    jaccard_similarity = char_metrics.Jaccard()
    err = jaccard_similarity.calculate(text1, text2) - expected_result
    assert -1e-5 < err < 1e-5


@pytest.mark.parametrize(
    "text1, text2, window, expected_result",
    [
        ("Word", "Wordy", 1, 0.19999999999999996),
        ("Word", "Wordy", 2, 0.25),
        ("Word", "Wordy", 3, 0.33333333333333337),
        ("Word", "Wordy", 10, 0.19999999999999996),
    ],
)
def test_jaccard_window(text1, text2, window, expected_result):
    jaccard_similarity = char_metrics.Jaccard()
    err = jaccard_similarity.calculate(text1, text2, ngrams=window) - expected_result
    assert -1e-5 < err < 1e-5


def test_jaccard_with_short_length():
    with pytest.raises(AssertionError):
        jaccard_similarity = char_metrics.Jaccard()
        jaccard_similarity.calculate("Word", "Wordy", ngrams=10, ignore=False)


@pytest.mark.parametrize(
    "text1, text2, expected_result",
    [
        (
            "The quick brown fox leapt at the lazy dog.",
            "The sprightly fox jumped at the indolent dog.",
            0.6916927,
        ),
        (
            "Sachin is the best batsman in the world.",
            "Gully Boy is the perfect rags to riches story.",
            0.29331264,
        ),
        ("He has insomnia.", "His productivity is coffee-fueled.", 0.38121146),
        ("She is bored to death.", "The sky looks so picturesque today.", 0.12955433),
    ],
)
def test_semantic_similarity(text1, text2, expected_result):
    semantic_similarity = char_metrics.SemanticSimilarity()
    assert math.isclose(
        semantic_similarity.calculate(text1, text2), expected_result, rel_tol=1e-6
    )
