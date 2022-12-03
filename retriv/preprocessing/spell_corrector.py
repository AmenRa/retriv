from functools import partial
from typing import Union

from hunspell import Hunspell

from ..utils.github_dowloader import download
from .utils import identity_function

BASE_URL = "https://github.com/wooorm/dictionaries/tree/main/dictionaries/"


# fmt: off
mapping = {
    "bg": "bg", "br": "br", "ca": "ca", "ca-valencia": "ca-valencia",
    "cs": "cs", "cy": "cy", "da": "da", "de": "de", "de-at": "de-at",
    "de-ch": "de-ch", "el": "el", "el-polyton": "el-polyton", "en": "en",
    "en-au": "en-au", "en-ca": "en-ca", "en-gb": "en-gb", "en-za": "en-za",
    "eo": "eo", "es": "es", "es-ar": "es-ar", "es-bo": "es-bo",
    "es-cl": "es-cl", "es-co": "es-co", "es-cr": "es-cr", "es-cu": "es-cu",
    "es-do": "es-do", "es-ec": "es-ec", "es-gt": "es-gt", "es-hn": "es-hn",
    "es-mx": "es-mx", "es-ni": "es-ni", "es-pa": "es-pa", "es-pe": "es-pe",
    "es-ph": "es-ph", "es-pr": "es-pr", "es-py": "es-py", "es-sv": "es-sv",
    "es-us": "es-us", "es-uy": "es-uy", "es-ve": "es-ve", "et": "et",
    "eu": "eu", "fa": "fa", "fo": "fo", "fr": "fr", "fur": "fur", "fy": "fy",
    "ga": "ga", "gd": "gd", "gl": "gl", "he": "he", "hr": "hr", "hu": "hu",
    "hy": "hy", "hyw": "hyw", "ia": "ia", "ie": "ie", "is": "is", "it": "it",
    "ka": "ka", "ko": "ko", "la": "la", "lb": "lb", "lt": "lt", "ltg": "ltg",
    "lv": "lv", "mk": "mk", "mn": "mn", "nb": "nb", "nds": "nds", "ne": "ne",
    "nl": "nl", "nn": "nn", "oc": "oc", "pl": "pl", "pt": "pt",
    "pt-pt": "pt-pt", "ro": "ro", "ru": "ru", "rw": "rw", "sk": "sk",
    "sl": "sl", "sr": "sr", "sr-latn": "sr-latn", "sv": "sv", "sv-fi": "sv-fi",
    "tk": "tk", "tlh": "tlh", "tlh-latn": "tlh-latn", "tr": "tr", "uk": "uk",
    "vi": "vi",
    #
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "finnish": "sv-fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hungarian": "hu",
    "italian": "it",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "spanish": "es",
    "swedish": "sv",
}
# fmt: on


def download_dictionary(lang: str):
    return download(BASE_URL + mapping[lang])


spell_corrector = None


def correct(x: str) -> str:
    try:
        if any(s in x for s in [" ", "-", "_", ".", "@"]):
            return x
        elif not spell_corrector.spell(x):
            return spell_corrector.suggest(x)[0].lower()
        else:
            return x
    except Exception:
        return x


def get_spell_corrector(lang: Union[str, bool]):
    if type(lang) is str:
        path = download_dictionary(lang)

        global spell_corrector
        spell_corrector = Hunspell("index", hunspell_data_dir=path)

        return partial(correct)
    elif lang is None:
        return identity_function
    else:
        raise (NotImplementedError)
