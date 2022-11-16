from typing import List

import nltk

nltk.download("stopwords", quiet=True)

supported_languages = {
    "arabic",
    "azerbaijani",
    "basque",
    "bengali",
    "catalan",
    "chinese",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hebrew",
    "hinglish",
    "hungarian",
    "indonesian",
    "italian",
    "kazakh",
    "nepali",
    "norwegian",
    "portuguese",
    "romanian",
    "russian",
    "slovene",
    "spanish",
    "swedish",
    "tajik",
    "turkish",
}


def get_stopwords(lang: str) -> List:
    assert (
        lang.lower() in supported_languages
    ), f"Stop-words for {lang} are not available."
    return nltk.corpus.stopwords.words(lang)
