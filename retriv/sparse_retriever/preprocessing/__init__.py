__all__ = [
    "lowercasing",
    "normalize_acronyms",
    "normalize_ampersand",
    "normalize_special_chars",
    "remove_punctuation",
    "strip_whitespaces",
    "get_stemmer",
    "get_tokenizer",
    "get_stopwords",
]


from typing import Callable, List, Set

from multipipe import Multipipe

from .normalization import (
    lowercasing,
    normalize_acronyms,
    normalize_ampersand,
    normalize_special_chars,
    remove_punctuation,
    strip_whitespaces,
)
from .stemmer import get_stemmer
from .stopwords import get_stopwords
from .tokenizer import get_tokenizer


def preprocessing(
    x: str,
    tokenizer: Callable,
    stopwords: Set[str],
    stemmer: Callable,
    do_lowercasing: bool,
    do_ampersand_normalization: bool,
    do_special_chars_normalization: bool,
    do_acronyms_normalization: bool,
    do_punctuation_removal: bool,
) -> List[str]:
    if do_lowercasing:
        x = lowercasing(x)
    if do_ampersand_normalization:
        x = normalize_ampersand(x)
    if do_special_chars_normalization:
        x = normalize_special_chars(x)
    if do_acronyms_normalization:
        x = normalize_acronyms(x)

    if tokenizer == str.split and do_punctuation_removal:
        x = remove_punctuation(x)
        x = strip_whitespaces(x)

    x = tokenizer(x)

    if tokenizer != str.split and do_punctuation_removal:
        x = [remove_punctuation(t) for t in x]
        x = [t for t in x if t]

    x = [t for t in x if t not in stopwords]

    return [stemmer(t) for t in x]


def preprocessing_multi(
    tokenizer: callable,
    stopwords: List[str],
    stemmer: callable,
    do_lowercasing: bool,
    do_ampersand_normalization: bool,
    do_special_chars_normalization: bool,
    do_acronyms_normalization: bool,
    do_punctuation_removal: bool,
):
    callables = []

    if do_lowercasing:
        callables.append(lowercasing)
    if do_ampersand_normalization:
        callables.append(normalize_ampersand)
    if do_special_chars_normalization:
        callables.append(normalize_special_chars)
    if do_acronyms_normalization:
        callables.append(normalize_acronyms)
    if tokenizer == str.split and do_punctuation_removal:
        callables.append(remove_punctuation)
        callables.append(strip_whitespaces)

    callables.append(tokenizer)

    if tokenizer != str.split and do_punctuation_removal:

        def rp(x):
            x = [remove_punctuation(t) for t in x]
            return [t for t in x if t]

        callables.append(rp)

    def sw(x):
        return [t for t in x if t not in stopwords]

    callables.append(sw)

    def stem(x):
        return [stemmer(t) for t in x]

    callables.append(stem)

    return Multipipe(callables)
