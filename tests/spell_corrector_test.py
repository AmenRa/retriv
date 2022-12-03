import pytest

from retriv.preprocessing.spell_corrector import get_spell_corrector

# FIXTURES =====================================================================


# TESTS ========================================================================
def test_spell_corrector():
    spell_corrector = get_spell_corrector("english")
    assert callable(spell_corrector)
    assert spell_corrector("hllo") == "hello"
    assert spell_corrector("mail.com") == "mail.com"
    assert spell_corrector("jhaegfsdjhb") == "jhaegfsdjhb"


def test_spell_corrector_none():
    spell_corrector = get_spell_corrector(None)
    assert callable(spell_corrector)
    assert spell_corrector("hllo") == "hllo"
    assert spell_corrector("mail.com") == "mail.com"
    assert spell_corrector("jhaegfsdjhb") == "jhaegfsdjhb"
