from functools import partial
from typing import Union

import nltk
from krovetzstemmer import Stemmer as KrovetzStemmer
from Stemmer import Stemmer as SnowballStemmer

from .utils import identity_function

nltk.download("rslp", quiet=True)

stemmers_dict = {
    "krovetz": partial(KrovetzStemmer()),
    "porter": partial(nltk.stem.PorterStemmer().stem),
    "lancaster": partial(nltk.stem.LancasterStemmer().stem),
    "arlstem": partial(nltk.stem.ARLSTem().stem),  # Arabic
    "arlstem2": partial(nltk.stem.ARLSTem2().stem),  # Arabic
    "cistem": partial(nltk.stem.Cistem().stem),  # German
    "isri": partial(nltk.stem.ISRIStemmer().stem),  # Arabic
    "rslp": partial(nltk.stem.RSLPStemmer().stem),  # Portuguese
    "arabic": partial(SnowballStemmer("arabic").stemWord),
    # "armenian": partial(SnowballStemmer("armenian").stemWord),
    "basque": partial(SnowballStemmer("basque").stemWord),
    "catalan": partial(SnowballStemmer("catalan").stemWord),
    "danish": partial(SnowballStemmer("danish").stemWord),
    "dutch": partial(SnowballStemmer("dutch").stemWord),
    "english": partial(nltk.stem.SnowballStemmer("english").stem),
    "finnish": partial(SnowballStemmer("finnish").stemWord),
    "french": partial(SnowballStemmer("french").stemWord),
    "german": partial(SnowballStemmer("german").stemWord),
    "greek": partial(SnowballStemmer("greek").stemWord),
    "hindi": partial(SnowballStemmer("hindi").stemWord),
    "hungarian": partial(SnowballStemmer("hungarian").stemWord),
    "indonesian": partial(SnowballStemmer("indonesian").stemWord),
    "irish": partial(SnowballStemmer("irish").stemWord),
    "italian": partial(SnowballStemmer("italian").stemWord),
    "lithuanian": partial(SnowballStemmer("lithuanian").stemWord),
    "nepali": partial(SnowballStemmer("nepali").stemWord),
    "norwegian": partial(SnowballStemmer("norwegian").stemWord),
    "portuguese": partial(SnowballStemmer("portuguese").stemWord),
    "romanian": partial(SnowballStemmer("romanian").stemWord),
    "russian": partial(SnowballStemmer("russian").stemWord),
    # "serbian": partial(SnowballStemmer("serbian").stemWord),
    "spanish": partial(SnowballStemmer("spanish").stemWord),
    "swedish": partial(SnowballStemmer("swedish").stemWord),
    "tamil": partial(SnowballStemmer("tamil").stemWord),
    "turkish": partial(SnowballStemmer("turkish").stemWord),
    # "yiddish": partial(SnowballStemmer("yiddish").stemWord),
    # "porter": partial(SnowballStemmer("porter").stemWord),
}


def krovetz_f(x: str) -> str:
    return stemmers_dict["krovetz"](x)


def porter_f(x: str) -> str:
    return stemmers_dict["porter"](x)


def lancaster_f(x: str) -> str:
    return stemmers_dict["lancaster"](x)


def arlstem_f(x: str) -> str:
    return stemmers_dict["arlstem"](x)


def arlstem2_f(x: str) -> str:
    return stemmers_dict["arlstem2"](x)


def cistem_f(x: str) -> str:
    return stemmers_dict["cistem"](x)


def isri_f(x: str) -> str:
    return stemmers_dict["isri"](x)


def rslp_f(x: str) -> str:
    return stemmers_dict["rslp"](x)


def arabic_f(x: str) -> str:
    return stemmers_dict["arabic"](x)


# def armenian_f(x: str) -> str:
#     return stemmers_dict["armenian"](x)


def basque_f(x: str) -> str:
    return stemmers_dict["basque"](x)


def catalan_f(x: str) -> str:
    return stemmers_dict["catalan"](x)


def danish_f(x: str) -> str:
    return stemmers_dict["danish"](x)


def dutch_f(x: str) -> str:
    return stemmers_dict["dutch"](x)


def english_f(x: str) -> str:
    return stemmers_dict["english"](x)


def finnish_f(x: str) -> str:
    return stemmers_dict["finnish"](x)


def french_f(x: str) -> str:
    return stemmers_dict["french"](x)


def german_f(x: str) -> str:
    return stemmers_dict["german"](x)


def greek_f(x: str) -> str:
    return stemmers_dict["greek"](x)


def hindi_f(x: str) -> str:
    return stemmers_dict["hindi"](x)


def hungarian_f(x: str) -> str:
    return stemmers_dict["hungarian"](x)


def indonesian_f(x: str) -> str:
    return stemmers_dict["indonesian"](x)


def irish_f(x: str) -> str:
    return stemmers_dict["irish"](x)


def italian_f(x: str) -> str:
    return stemmers_dict["italian"](x)


def lithuanian_f(x: str) -> str:
    return stemmers_dict["lithuanian"](x)


def nepali_f(x: str) -> str:
    return stemmers_dict["nepali"](x)


def norwegian_f(x: str) -> str:
    return stemmers_dict["norwegian"](x)


def portuguese_f(x: str) -> str:
    return stemmers_dict["portuguese"](x)


def romanian_f(x: str) -> str:
    return stemmers_dict["romanian"](x)


def russian_f(x: str) -> str:
    return stemmers_dict["russian"](x)


# def serbian_f(x: str) -> str:
#     return stemmers_dict["serbian"](x)


def spanish_f(x: str) -> str:
    return stemmers_dict["spanish"](x)


def swedish_f(x: str) -> str:
    return stemmers_dict["swedish"](x)


def tamil_f(x: str) -> str:
    return stemmers_dict["tamil"](x)


def turkish_f(x: str) -> str:
    return stemmers_dict["turkish"](x)


# def yiddish_f(x: str) -> str:
#     return stemmers_dict["yiddish"](x)


stemmers_f_dict = {
    "krovetz": krovetz_f,
    "porter": porter_f,
    "lancaster": lancaster_f,
    "arlstem": arlstem_f,
    "arlstem2": arlstem2_f,
    "cistem": cistem_f,
    "isri": isri_f,
    "rslp": rslp_f,
    "arabic": arabic_f,
    # "armenian": armenian_f,
    "basque": basque_f,
    "catalan": catalan_f,
    "danish": danish_f,
    "dutch": dutch_f,
    "english": english_f,
    "finnish": finnish_f,
    "french": french_f,
    "german": german_f,
    "greek": greek_f,
    "hindi": hindi_f,
    "hungarian": hungarian_f,
    "indonesian": indonesian_f,
    "irish": irish_f,
    "italian": italian_f,
    "lithuanian": lithuanian_f,
    "nepali": nepali_f,
    "norwegian": norwegian_f,
    "portuguese": portuguese_f,
    "romanian": romanian_f,
    "russian": russian_f,
    # "serbian": serbian_f,
    "spanish": spanish_f,
    "swedish": swedish_f,
    "tamil": tamil_f,
    "turkish": turkish_f,
    # "yiddish": yiddish_f,
}


def _get_stemmer(stemmer: str) -> callable:
    assert (
        stemmer.lower() in stemmers_f_dict
    ), f"Stemmer {stemmer} not supported."
    return stemmers_f_dict[stemmer.lower()]


def get_stemmer(stemmer: Union[str, callable, bool]) -> callable:
    if type(stemmer) is str:
        return _get_stemmer(stemmer)
    elif callable(stemmer):
        return stemmer
    elif stemmer is None:
        return identity_function
    else:
        raise (NotImplementedError)
