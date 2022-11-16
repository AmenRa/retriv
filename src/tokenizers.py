from typing import Callable

from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize

tokenizers_dict = {
    "whitespace": str.split,
    "word": word_tokenize,
    "wordpunct": wordpunct_tokenize,
    "sent": sent_tokenize,
}


def get_tokenizer(tokenizer: str) -> Callable:
    assert (
        tokenizer.lower() in tokenizers_dict
    ), f"Tokenizer {tokenizer} not supported."
    return tokenizers_dict[tokenizer.lower()]
