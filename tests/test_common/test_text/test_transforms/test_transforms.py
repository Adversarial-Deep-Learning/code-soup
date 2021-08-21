import random

import pytest

from code_soup.common.text.transforms import transforms


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "T winkle twinkle little star."),
        ("Hey, this is so fascinating!", "H ey, this is so fascinating!"),
        ("The earthen pot has cold water.", "The earthen pot has cold w ater."),
    ],
)
def test_add_space(text, expected_result):
    random.seed(42)
    tfms = transforms.AddChar(extractor="RandomWordExtractor", char_perturb=False)
    assert tfms(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "xTwinkle twinkle little star."),
        ("Hey, this is so fascinating!", "xHey, this is so fascinating!"),
        ("The earthen pot has cold water.", "The earthen pot has cold awater."),
    ],
)
def test_add_char(text, expected_result):
    random.seed(42)
    tfms = transforms.AddChar(extractor="RandomWordExtractor", char_perturb=True)
    assert tfms(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "Tiwnkle twinkle little star."),
        ("Hey, this is so fascinating!", "Hye, this is so fascinating!"),
        ("The earthen pot has cold water.", "The earthen pot has cold wtaer."),
    ],
)
def test_shuffle_char(text, expected_result):
    random.seed(42)
    tfms = transforms.ShuffleChar(extractor="RandomWordExtractor", mid=False)
    assert tfms(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "Tiklnwe twinkle little star."),
        ("Hey, this is so fascinating!", "Hye, this is so fascinating!"),
        ("The earthen pot has cold water.", "The earthen pot has cold wetra."),
    ],
)
def test_shuffle_char_2(text, expected_result):
    random.seed(42)
    tfms = transforms.ShuffleChar(extractor="RandomWordExtractor", mid=True)
    assert tfms(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "Tinkle twinkle little star."),
        ("Hey, this is so fascinating!", "Hy, this is so fascinating!"),
        ("The earthen pot has cold water.", "The earthen pot has cold wter."),
    ],
)
def test_delete_char(text, expected_result):
    random.seed(42)
    tfms = transforms.DeleteChar(extractor="RandomWordExtractor")
    assert tfms(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "Fwjnkke twinkle little star."),
        ("Hey, this is so fascinating!", "Ueg, this is so fascinating!"),
        ("The earthen pot has cold water.", "The earthen pot has cold sater."),
    ],
)
def test_typo_char(text, expected_result):
    random.seed(42)
    tfms = transforms.TypoChar(extractor="RandomWordExtractor", probability=0.3)
    assert tfms(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "Ｔ𑜊ӏ𝖓𝓀1𝖾 twinkle little star."),
        ("Hey, this is so fascinating!", "H́e̐y̒,̂ this is so fascinating!"),
        ("The earthen pot has cold water.", "The earthen pot has cold w̐a̅t̒êr̅.̒"),
    ],
)
def test_visually_similar_char(text, expected_result):
    random.seed(42)
    tfms = transforms.VisuallySimilarChar(extractor="RandomWordExtractor", seed=None)
    assert tfms(text) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Twinkle twinkle little star.", "R winkle tknwlie little 𝒔𝘵𝖆𝘳."),
        ("Hey, this is so fascinating!", "Y̐ ey, this is so fitcgaasnin!"),
        ("The earthen pot has cold water.", "The earthen oor has cold 𝚠 ater."),
    ],
)
def test_compose_transforms(text, expected_result):
    random.seed(42)
    tfms = transforms.Compose(
        [
            transforms.AddChar(),
            transforms.ShuffleChar("RandomWordExtractor", True),
            transforms.VisuallySimilarChar(),
            transforms.TypoChar("RandomWordExtractor", probability=0.5),
        ]
    )
    assert tfms(text) == expected_result
