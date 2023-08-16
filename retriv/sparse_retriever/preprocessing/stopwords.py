from typing import List, Set, Union

import nltk

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


def _get_stopwords(lang: str) -> List[str]:
    nltk.download("stopwords", quiet=True)
    assert (
        lang.lower() in supported_languages
    ), f"Stop-words for {lang.capitalize()} are not available."
    return nltk.corpus.stopwords.words(lang)


def get_stopwords(sw_list: Union[str, List[str], Set[str], bool]) -> List[str]:
    if isinstance(sw_list, str):
        return _get_stopwords(sw_list)
    elif type(sw_list) is list and all(isinstance(x, str) for x in sw_list):
        return sw_list
    elif type(sw_list) is set:
        return list(sw_list)
    elif sw_list is None:
        return []
    else:
        raise (NotImplementedError)
