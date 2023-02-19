import pytest

from retriv.sparse_retriever.preprocessing.stopwords import get_stopwords


# FIXTURES =====================================================================
@pytest.fixture
def supported_languages():
    # fmt: off
    return [
        "arabic", "azerbaijani", "basque", "catalan", "bengali", "chinese",
        "danish", "dutch", "english", "finnish", "french", "german", "greek",
        "hebrew", "hinglish", "hungarian", "indonesian", "italian", "kazakh",
        "nepali", "norwegian", "portuguese", "romanian", "russian", "slovene",
        "spanish", "swedish", "tajik", "turkish",
    ]
    # fmt: on


@pytest.fixture
def sw_list():
    return ["a", "the"]


@pytest.fixture
def sw_set():
    return {"a", "the"}


# TESTS ========================================================================
def test_get_stopwords_from_lang(supported_languages):
    for lang in supported_languages:
        assert type(get_stopwords(lang)) == list
        assert len(get_stopwords(lang)) > 0


def test_get_stopwords_from_lang_fails():
    with pytest.raises(Exception):
        get_stopwords("foobar")


def test_get_stopwords_from_list(sw_list):
    assert type(get_stopwords(sw_list)) == list
    assert set(get_stopwords(sw_list)) == {"a", "the"}
    assert len(get_stopwords(sw_list)) > 0


def test_get_stopwords_from_set(sw_set):
    assert type(get_stopwords(sw_set)) == list
    assert set(get_stopwords(sw_set)) == {"a", "the"}
    assert len(get_stopwords(sw_set)) > 0


def test_get_stopwords_none():
    assert type(get_stopwords(None)) == list
    assert len(get_stopwords(None)) == 0
