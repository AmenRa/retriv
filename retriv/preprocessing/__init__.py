__all__ = [
    "lowercasing",
    "normalize_acronyms",
    "normalize_ampersand",
    "normalize_special_chars",
    "remove_punctuation",
    "strip_whitespaces",
    "get_spell_corrector",
    "get_stemmer",
    "get_tokenizer",
    "get_stopwords",
]


from multiprocessing import Pool
from typing import Callable, Generator, Iterable, List, Set

from .normalization import (
    lowercasing,
    normalize_acronyms,
    normalize_ampersand,
    normalize_special_chars,
    remove_punctuation,
    strip_whitespaces,
)
from .spell_corrector import get_spell_corrector
from .stemmer import get_stemmer
from .stopwords import get_stopwords
from .tokenizer import get_tokenizer


def preprocessing(
    x: str,
    tokenizer: Callable,
    stopwords: Set[str],
    stemmer: Callable,
    spell_corrector: Callable,
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

    x = [spell_corrector(t) for t in x]
    x = [t for t in x if t not in stopwords]

    return [stemmer(t) for t in x]


def _preprocessing(x):
    return preprocessing(*x)


def prepare_inputs(
    collection: Iterable,
    tokenizer: callable,
    stopwords: List[str],
    stemmer: callable,
    spell_corrector: callable,
    do_lowercasing: bool,
    do_ampersand_normalization: bool,
    do_special_chars_normalization: bool,
    do_acronyms_normalization: bool,
    do_punctuation_removal: bool,
) -> Generator:
    for doc in collection:
        yield (
            doc,
            tokenizer,
            stopwords,
            stemmer,
            spell_corrector,
            do_lowercasing,
            do_ampersand_normalization,
            do_special_chars_normalization,
            do_acronyms_normalization,
            do_punctuation_removal,
        )


def multi_preprocessing(
    collection: Iterable,
    tokenizer: callable,
    stopwords: List[str],
    stemmer: callable,
    spell_corrector: callable,
    do_lowercasing: bool,
    do_ampersand_normalization: bool,
    do_special_chars_normalization: bool,
    do_acronyms_normalization: bool,
    do_punctuation_removal: bool,
    n_threads: int,
) -> Generator:
    inputs = prepare_inputs(
        collection,
        tokenizer=tokenizer,
        stopwords=stopwords,
        stemmer=stemmer,
        spell_corrector=spell_corrector,
        do_lowercasing=do_lowercasing,
        do_ampersand_normalization=do_ampersand_normalization,
        do_special_chars_normalization=do_special_chars_normalization,
        do_acronyms_normalization=do_acronyms_normalization,
        do_punctuation_removal=do_punctuation_removal,
    )

    with Pool(n_threads) as p:
        yield from p.imap(_preprocessing, inputs, chunksize=1_000)
