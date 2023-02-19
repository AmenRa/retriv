import pytest

from retriv.sparse_retriever.preprocessing.stemmer import get_stemmer


# FIXTURES =====================================================================
@pytest.fixture
def supported_stemmers():
    # fmt: off
    return [
        "krovetz", "porter", "lancaster", "arlstem", "arlstem2", "cistem",
        "isri", "arabic",
        "basque", "catalan", "danish", "dutch", "english", "finnish", "french", "german", "greek", "hindi", "hungarian", "indonesian", "irish", "italian", "lithuanian", "nepali", "norwegian", "portuguese", "romanian", "russian",
        "spanish", "swedish", "tamil", "turkish",
        ]
    # fmt: on


# TESTS ========================================================================
def test_get_stemmer(supported_stemmers):
    for stemmer in supported_stemmers:
        assert callable(get_stemmer(stemmer))


def test_get_stemmer_fails():
    with pytest.raises(Exception):
        get_stemmer("foobar")


def test_get_stemmer_callable():
    assert callable(get_stemmer(lambda x: x))


def test_get_stemmer_none():
    assert callable(get_stemmer(None))
    assert get_stemmer(None)("incredible") == "incredible"
