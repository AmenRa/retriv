from functools import partial
from typing import Callable

import nltk
from krovetzstemmer import Stemmer
from nltk.stem import (
    ARLSTem,
    ARLSTem2,
    Cistem,
    ISRIStemmer,
    LancasterStemmer,
    PorterStemmer,
    RSLPStemmer,
    SnowballStemmer,
)

nltk.download("rslp", quiet=True)

stemmers_dict = {
    "krovetz": partial(Stemmer()),
    "porter": partial(PorterStemmer().stem),
    "lancaster": partial(LancasterStemmer().stem),
    "arlstem": partial(ARLSTem().stem),  # Arabic
    "arlstem2": partial(ARLSTem2().stem),  # Arabic
    "cistem": partial(Cistem().stem),  # German
    "isri": partial(ISRIStemmer().stem),  # Arabic
    "rslp": partial(RSLPStemmer().stem),  # Portuguese
    "arabic": partial(SnowballStemmer("arabic").stem),
    "snowball-arabic": partial(SnowballStemmer("arabic").stem),
    "danish": partial(SnowballStemmer("danish").stem),
    "snowball-danish": partial(SnowballStemmer("danish").stem),
    "dutch": partial(SnowballStemmer("dutch").stem),
    "snowball-dutch": partial(SnowballStemmer("dutch").stem),
    "english": partial(SnowballStemmer("english").stem),
    "snowball-english": partial(SnowballStemmer("english").stem),
    "finnish": partial(SnowballStemmer("finnish").stem),
    "snowball-finnish": partial(SnowballStemmer("finnish").stem),
    "french": partial(SnowballStemmer("french").stem),
    "snowball-french": partial(SnowballStemmer("french").stem),
    "german": partial(SnowballStemmer("german").stem),
    "snowball-german": partial(SnowballStemmer("german").stem),
    "hungarian": partial(SnowballStemmer("hungarian").stem),
    "snowball-hungarian": partial(SnowballStemmer("hungarian").stem),
    "italian": partial(SnowballStemmer("italian").stem),
    "snowball-italian": partial(SnowballStemmer("italian").stem),
    "norwegian": partial(SnowballStemmer("norwegian").stem),
    "snowball-norwegian": partial(SnowballStemmer("norwegian").stem),
    "porter": partial(SnowballStemmer("porter").stem),
    "snowball-porter": partial(SnowballStemmer("porter").stem),
    "portuguese": partial(SnowballStemmer("portuguese").stem),
    "snowball-portuguese": partial(SnowballStemmer("portuguese").stem),
    "romanian": partial(SnowballStemmer("romanian").stem),
    "snowball-romanian": partial(SnowballStemmer("romanian").stem),
    "russian": partial(SnowballStemmer("russian").stem),
    "snowball-russian": partial(SnowballStemmer("russian").stem),
    "spanish": partial(SnowballStemmer("spanish").stem),
    "snowball-spanish": partial(SnowballStemmer("spanish").stem),
    "swedish": partial(SnowballStemmer("swedish").stem),
    "snowball-swedish": partial(SnowballStemmer("swedish").stem),
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


def hungarian_f(x: str) -> str:
    return stemmers_dict["hungarian"](x)


def italian_f(x: str) -> str:
    return stemmers_dict["italian"](x)


def norwegian_f(x: str) -> str:
    return stemmers_dict["norwegian"](x)


def porter_f(x: str) -> str:
    return stemmers_dict["porter"](x)


def portuguese_f(x: str) -> str:
    return stemmers_dict["portuguese"](x)


def romanian_f(x: str) -> str:
    return stemmers_dict["romanian"](x)


def russian_f(x: str) -> str:
    return stemmers_dict["russian"](x)


def spanish_f(x: str) -> str:
    return stemmers_dict["spanish"](x)


def swedish_f(x: str) -> str:
    return stemmers_dict["swedish"](x)


def wordnet_f(x: str) -> str:
    return stemmers_dict["wordnet"](x)


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
    "danish": danish_f,
    "dutch": dutch_f,
    "english": english_f,
    "finnish": finnish_f,
    "french": french_f,
    "german": german_f,
    "hungarian": hungarian_f,
    "italian": italian_f,
    "norwegian": norwegian_f,
    "porter": porter_f,
    "portuguese": portuguese_f,
    "romanian": romanian_f,
    "russian": russian_f,
    "spanish": spanish_f,
    "swedish": swedish_f,
    "wordnet": wordnet_f,
}


def get_stemmer(stemmer: str) -> Callable:
    assert (
        stemmer.lower() in stemmers_f_dict
    ), f"Stemmer {stemmer} not supported."
    return stemmers_f_dict[stemmer.lower()]
