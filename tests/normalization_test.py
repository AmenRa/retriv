import pytest

from retriv.preprocessing.normalization import (
    lowercasing,
    normalize_acronyms,
    normalize_ampersand,
    normalize_special_chars,
    remove_punctuation,
    strip_whitespaces,
)


# TESTS ========================================================================
def test_lowercasing():
    assert lowercasing("hEllO") == "hello"


def test_normalize_ampersand():
    assert normalize_ampersand("black&sabbath") == "black and sabbath"


def test_normalize_special_chars():
    assert normalize_special_chars("‘’") == "''"


def test_normalize_acronyms():
    assert normalize_acronyms("a.b.c.") == "abc"
    assert normalize_acronyms("foo.bar") == "foo.bar"
    assert normalize_acronyms("a.b@hello.com") == "a.b@hello.com"


def test_remove_punctuation():
    assert remove_punctuation("foo.bar?") == "foo bar "
    # assert remove_punctuation("a.b@hello.com") == "a.b@hello.com"


def test_strip_whitespaces():
    assert strip_whitespaces(" hello   world  ") == "hello world"
