import pytest

from retriv.dense_retriever.encoder import Encoder


# FIXTURES =====================================================================
@pytest.fixture
def encoder():
    return Encoder(model="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def texts():
    return [
        "Generals gathered in their masses",
        "Just like witches at black masses",
        "Evil minds that plot destruction",
        "Sorcerer of death's construction",
    ]


# TESTS ========================================================================
def test_call(encoder, texts):
    embeddings = encoder(texts)
    assert embeddings.shape[0] == 4
    assert embeddings.shape[1] == encoder.embedding_dim
