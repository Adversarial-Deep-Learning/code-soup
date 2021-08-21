import random

import pytest

from code_soup.common.text.transforms import paraphrase

LENGTH_CONTRACTION_EXAMPLE = "had"
NOT_A_CONTRACTION_EXAMPLE = "Shes"


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("I had", "I'd"),
        ("I would have", "I'd've"),
        ("Rohan is", "Rohan's"),
        ("I am", "I'm"),
        ("They are", "They're"),
    ],
)
def test_paraphrase_contraction(text, expected_result):
    random.seed(42)
    contractions_apply = paraphrase.ContractParaphrases()
    assert contractions_apply.apply(text, reverse=False, ignore=True) == expected_result


def test_paraphrase_contraction_length_assertion():
    contractions_apply = paraphrase.ContractParaphrases()
    with pytest.raises(AssertionError):
        contractions_apply.apply(
            LENGTH_CONTRACTION_EXAMPLE, reverse=False, ignore=False
        )


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("He's", "He has"),
        ("I'd've", "I would have"),
        ("They won't", "They will not"),
        ("Let's", "let us"),
    ],
)
def test_paraphrase_expansion(text, expected_result):
    random.seed(41)

    contractions_apply = paraphrase.ContractParaphrases()
    assert contractions_apply.apply(text, reverse=True, ignore=True) == expected_result


def test_paraphrase_not_a_contraction_assertion():
    contractions_apply = paraphrase.ContractParaphrases()
    with pytest.raises(AssertionError):
        assert contractions_apply.apply(
            NOT_A_CONTRACTION_EXAMPLE, reverse=True, ignore=False
        )
