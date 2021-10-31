def words_from_text(s, words_to_ignore=[]):
    homos = set(
        [
            "˗",
            "৭",
            "Ȣ",
            "𝟕",
            "б",
            "Ƽ",
            "Ꮞ",
            "Ʒ",
            "ᒿ",
            "l",
            "O",
            "`",
            "ɑ",
            "Ь",
            "ϲ",
            "ԁ",
            "е",
            "𝚏",
            "ɡ",
            "հ",
            "і",
            "ϳ",
            "𝒌",
            "ⅼ",
            "ｍ",
            "ո",
            "о",
            "р",
            "ԛ",
            "ⲅ",
            "ѕ",
            "𝚝",
            "ս",
            "ѵ",
            "ԝ",
            "×",
            "у",
            "ᴢ",
        ]
    )
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum() or c in homos:
            word += c
        elif c in "'-_*@" and len(word) > 0:
            # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return words

def is_one_word(word):
    return len(words_from_text(word)) == 1