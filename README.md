<div align="center">
  <img src="https://repository-images.githubusercontent.com/566840861/ce7eeed0-7454-4aff-9073-235a83eeb6e7">
</div>

<p align="center">
  <!-- Python -->
  <a href="https://www.python.org" alt="Python">
      <img src="https://badges.aleen42.com/src/python.svg" />
  </a>
  <!-- Version -->
  <a href="https://badge.fury.io/py/retriv"><img src="https://badge.fury.io/py/retriv.svg" alt="PyPI version" height="18"></a>
  <!-- Docs -->
  <!-- <a href="https://amenra.github.io/retriv"><img src="https://img.shields.io/badge/docs-passing-<COLOR>.svg" alt="Documentation Status"></a> -->
  <!-- Black -->
  <a href="https://github.com/psf/black" alt="Code style: black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
  </a>
  <!-- License -->
  <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <!-- Google Colab -->
  <!-- <a href="https://colab.research.google.com/github/AmenRa/retriv/blob/master/notebooks/1_overview.ipynb"> -->
      <!-- <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> -->
  </a>
</p>

## 🔥 News
- [February 18, 2023] `retriv` 0.2.0 is out!  
This adds support for Dense and Hybrid Retrieval.
Dense Retrieval leverages the semantic similarity of the queries' and documents' vector representations, which can be computed directly by `retriv` or imported from other sources.
Hybrid Retrieval mix traditional retrieval, informally called Sparse Retrieval,  and Dense Retrieval results to further improve retrieval effectiveness.
As the library was almost completely redone, indices built with previous versions are no longer supported.

## ⚡️ Introduction

[retriv](https://github.com/AmenRa/retriv) is a user-friendly and efficient [search engine](https://en.wikipedia.org/wiki/Search_engine) implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) supporting Sparse (traditional search with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf)), Dense ([semantic search](https://en.wikipedia.org/wiki/Semantic_search)) and Hybrid retrieval (a mix of Sparse and Dense Retrieval).
It allows you to build a search engine in a __single line of code__.

[retriv](https://github.com/AmenRa/retriv) is built upon [Numba](https://github.com/numba/numba) for high-speed [vector operations](https://en.wikipedia.org/wiki/Automatic_vectorization) and [automatic parallelization](https://en.wikipedia.org/wiki/Automatic_parallelization), [PyTorch](https://pytorch.org) and [Transformers](https://huggingface.co/docs/transformers/index) for easy access and usage of [Transformer-based Language Models](https://web.stanford.edu/~jurafsky/slp3/10.pdf), and [Faiss](https://github.com/facebookresearch/faiss) for approximate [nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search).
In addition, it provides automatic tuning functionalities to allow you to tune its internal components with minimal intervention.


## ✨ Main Features

### Retrievers
- [Sparse Retriever](https://github.com/AmenRa/retriv/blob/main/docs/sparse_retriever.md): standard searcher based on lexical matching. 
[retriv](https://github.com/AmenRa/retriv) implements [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) as its main retrieval model.
[TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) is also supported for educational purposes.
The sparse retriever comes armed with multiple [stemmers](https://en.wikipedia.org/wiki/Stemming), [tokenizers](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization), and [stop-word](https://en.wikipedia.org/wiki/Stop_word) lists, for multiple languages.
Click [here](https://github.com/AmenRa/retriv/blob/main/docs/sparse_retriever.md) to learn more.
- [Dense Retriever](https://github.com/AmenRa/retriv/blob/main/docs/dense_retriever.md): a dense retriever is a retrieval model that performs [semantic search](https://en.wikipedia.org/wiki/Semantic_search). 
Click [here](https://github.com/AmenRa/retriv/blob/main/docs/dense_retriever.md) to learn more.
- [Hybrid Retriever](https://github.com/AmenRa/retriv/blob/main/docs/hybrid_retriever.md): an hybrid retriever is a retrieval model built on top of a sparse and a dense retriever.
Click [here](https://github.com/AmenRa/retriv/blob/main/docs/hybrid_retriever.md) to learn more.

### Unified Search Interface
All the supported retrievers share the same search interface:
- [search](#search): standard search functionality, what you expect by a search engine.
- [msearch](#multi-search): computes the results for multiple queries at once.
It leverages [automatic parallelization](https://en.wikipedia.org/wiki/Automatic_parallelization) whenever possible.
- [bsearch](#batch-search): similar to [msearch](#multi-search) but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format. [bsearch](#batch-search) is handy for computing results for hundreds of thousands or even millions of queries without hogging your RAM. Pre-computed results can be leveraged for negative sampling during the training of [Neural Models](https://en.wikipedia.org/wiki/Artificial_neural_network) for [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval).

### AutoTune
[retriv](https://github.com/AmenRa/retriv) automatically tunes [Faiss](https://github.com/facebookresearch/faiss) configuration for approximate nearest neighbors search by leveraging [AutoFaiss](https://github.com/criteo/autofaiss) to guarantee 10ms response time based on your available hardware.
Moreover, it offers an automatic tuning functionality for [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)'s parameters, which require minimal user intervention.
Under the hood, [retriv](https://github.com/AmenRa/retriv) leverages [Optuna](https://optuna.org), a [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) framework, and [ranx](https://github.com/AmenRa/ranx), an [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval) evaluation library, to test several parameter configurations for [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and choose the best one.
Finally, it can automatically balance the importance of lexical and semantic relevance scores computed by the [Hybrid Retriever](https://github.com/AmenRa/retriv/blob/main/docs/hybrid_retriever.md) to maximize retrieval effectiveness.

## 📚 Documentation

- [Sparse Retriever](https://github.com/AmenRa/retriv/blob/main/docs/sparse_retriever.md)
- [Dense Retriever](https://github.com/AmenRa/retriv/blob/main/docs/dense_retriever.md)
- [Hybrid Retriever](https://github.com/AmenRa/retriv/blob/main/docs/hybrid_retriever.md)
- [Text Pre-Processing](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md)
- [FAQ](https://github.com/AmenRa/retriv/blob/main/docs/faq.md)

## 🔌 Requirements
```
python>=3.8
```

## 💾 Installation
```bash
pip install retriv
```

## 💡 Minimal Working Example

```python
# Note: SearchEngine is an alias for the SparseRetriever
from retriv import SearchEngine

collection = [
  {"id": "doc_1", "text": "Generals gathered in their masses"},
  {"id": "doc_2", "text": "Just like witches at black masses"},
  {"id": "doc_3", "text": "Evil minds that plot destruction"},
  {"id": "doc_4", "text": "Sorcerer of death's construction"},
]

se = SearchEngine("new-index").index(collection)

se.search("witches masses")
```
Output:
```python
[
  {
    "id": "doc_2",
    "text": "Just like witches at black masses",
    "score": 1.7536403
  },
  {
    "id": "doc_1",
    "text": "Generals gathered in their masses",
    "score": 0.6931472
  }
]
```






## 🎁 Feature Requests
Would you like to see other features implemented? Please, open a [feature request](https://github.com/AmenRa/retriv/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature+Request%5D+title).


## 🤘 Want to contribute?
Would you like to contribute? Please, drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20retriv).


## 📄 License
[retriv](https://github.com/AmenRa/retriv) is an open-sourced software licensed under the [MIT license](LICENSE).
