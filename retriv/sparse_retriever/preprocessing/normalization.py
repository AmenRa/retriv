import re
import string

from unidecode import unidecode


def lowercasing(x: str) -> str:
    return x.lower()


def normalize_ampersand(x: str) -> str:
    return x.replace("&", " and ")


def normalize_diacritics(x: str) -> str:
    return unidecode(x)


def normalize_special_chars(x: str) -> str:
    special_chars_trans = dict(
        [(ord(x), ord(y)) for x, y in zip("‘’´“”–-", "'''\"\"--")]
    )
    return x.translate(special_chars_trans)


def normalize_acronyms(x: str) -> str:
    return re.sub(r"\.(?!(\S[^. ])|\d)", "", x)


def remove_punctuation(x: str) -> str:
    translator = str.maketrans(
        string.punctuation, " " * len(string.punctuation)
    )
    return x.translate(translator)


def strip_whitespaces(x: str) -> str:
    x = x.strip()

    while "  " in x:
        x = x.replace("  ", " ")

    return x
