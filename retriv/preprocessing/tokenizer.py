from typing import Union

import nltk

from .utils import identity_function

tokenizers_dict = {
    "whitespace": str.split,
    "word": nltk.tokenize.word_tokenize,
    "wordpunct": nltk.tokenize.wordpunct_tokenize,
    "sent": nltk.tokenize.sent_tokenize,
}

nltk.download("punkt", quiet=True)


def _get_tokenizer(tokenizer: str) -> callable:
    assert (
        tokenizer.lower() in tokenizers_dict
    ), f"Tokenizer {tokenizer} not supported."
    return tokenizers_dict[tokenizer.lower()]


def get_tokenizer(tokenizer: Union[str, callable, bool]) -> callable:
    if type(tokenizer) is str:
        return _get_tokenizer(tokenizer)
    elif callable(tokenizer):
        return tokenizer
    elif tokenizer is None:
        return identity_function
    else:
        raise (NotImplementedError)
