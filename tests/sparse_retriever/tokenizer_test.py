import pytest

from retriv.sparse_retriever.preprocessing.tokenizer import get_tokenizer


# FIXTURES =====================================================================
@pytest.fixture
def supported_tokenizers():
    return ["whitespace", "word", "wordpunct", "sent"]


# TESTS ========================================================================
def test_get_tokenizer(supported_tokenizers):
    for tokenizer in supported_tokenizers:
        assert callable(get_tokenizer(tokenizer))


def test_get_tokenizer_fails():
    with pytest.raises(Exception):
        get_tokenizer("foobar")


def test_get_tokenizer_callable():
    assert callable(get_tokenizer(lambda x: x))


def test_get_tokenizer_none():
    assert callable(get_tokenizer(None))
    assert get_tokenizer(None)("black sabbath") == "black sabbath"
