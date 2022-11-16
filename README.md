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

## ⚡️ Introduction

[retriv](https://github.com/AmenRa/retriv) is a fast search engine implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), leveraging [Numba](https://github.com/numba/numba) for high-speed [vector operations](https://en.wikipedia.org/wiki/Automatic_vectorization) and [automatic parallelization](https://en.wikipedia.org/wiki/Automatic_parallelization).
It offers a user-friendly interface to index and search your document collection and allows you to automatically tune the underling retrieval model, [BM25](https://en.wikipedia.org/wiki/Okapi_BM25).


## ✨ Features

### Stemmers
[Stemmers](https://en.wikipedia.org/wiki/Stemming) reduce words to their word stem, base or root form.  
[retriv](https://github.com/AmenRa/retriv) supports the following stemmers:
- [snowball](https://www.nltk.org/api/nltk.stem.snowball.html) (default)  
The following languages are supported by Snowball Stemmer: 
Arabic, Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish.
To select your preferred language simply use `<language>` .
- [arlstem](https://www.nltk.org/api/nltk.stem.arlstem.html) (Arabic)
- [arlstem2](https://www.nltk.org/api/nltk.stem.arlstem2.html) (Arabic)
- [cistem](https://www.nltk.org/api/nltk.stem.cistem.html) (German)
- [isri](https://www.nltk.org/api/nltk.stem.isri.html) (Arabic)
- [krovetz](https://dl.acm.org/doi/10.1145/160688.160718) (English)
- [lancaster](https://www.nltk.org/api/nltk.stem.lancaster.html) (English)
- [porter](https://www.nltk.org/api/nltk.stem.porter.html) (English)
- [rslp](https://www.nltk.org/api/nltk.stem.rslp.html) (Portuguese)

### Tokenizers

[Tokenizers](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) divide a string into smaller units, such as words.  
[retriv](https://github.com/AmenRa/retriv) supports the following tokenizers:
- [whitespace](https://www.nltk.org/api/nltk.tokenize.html)
- [word](https://www.nltk.org/api/nltk.tokenize.html)
- [wordpunct](https://www.nltk.org/api/nltk.tokenize.html)
- [sent](https://www.nltk.org/api/nltk.tokenize.html)

### Stop-word Lists

[retriv](https://github.com/AmenRa/retriv) supports [stop-word](https://en.wikipedia.org/wiki/Stop_word) lists for the following languages: Arabic, Azerbaijani, Basque, Bengali, Catalan, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hinglish, Hungarian, Indonesian, Italian, Kazakh, Nepali, Norwegian, Portuguese, Romanian, Russian, Slovene, Spanish, Swedish, Tajik, and Turkish.

### Retrieval Models
- [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- More coming soon...

### AutoTune

[retriv](https://github.com/AmenRa/retriv) supports an automatic tuning functionality that allows you to tune [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)'s parameters with a single function call.
Under the hood, [retriv](https://github.com/AmenRa/retriv) leverage [Optuna](https://optuna.org), a [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) framework, and [ranx](https://github.com/AmenRa/ranx), an [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval) evaluation library, to test several parameter configurations for [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and choose the best one.

## 🔌 Installation
```bash
pip install retriv
```

## 💡 Usage

### Create index
```python
from retriv import SearchEngine

collection = [
  {"id": "doc_1", "contents": "Generals gathered in their masses"},
  {"id": "doc_2", "contents": "Just like witches at black masses"},
  {"id": "doc_3", "contents": "Evil minds that plot destruction"},
  {"id": "doc_4", "contents": "Sorcerer of death's construction"},
]

se = SearchEngine(
  index_name="new-index",  # Default value
  min_term_freq=1,         # Default value
  tokenizer="whitespace",  # Default value
  stemmer="english",       # Default value (Snowball English stemmer)
  sw_list="english",       # Default value
)

se.index(
  collection=collection,
  show_progress=True,     # Default value
)
```

Alternatively, you can index a document collection from a JSONl, CSV, or TSV file.
CSV and TSV files must have a header.
Use the `callback` parameter to pass a function for converting your documents in the format supported by [retriv](https://github.com/AmenRa/retriv).

```python
se = SearchEngine("index-from-file")
se.index_file(
  path="path/to/collection",
  show_progress=True,     # Default value
  callback=None,          # Default value
)
```

### Search
```python
se.search(
  query="witches masses",
  return_docs=True,  # Default value
  b=0.75,            # Default value, BM25 parameter
  k1=1.2,            # Default value, BM25 parameter
  n_res=100,         # Default value, number of results
)
```
Output:
```
[
  {
    "id": "doc_2",
    "contents": "Just like witches at black masses",
    "score": 1.7536403
  },
  {
    "id": "doc_1",
    "contents": "Generals gathered in their masses",
    "score": 0.6931472
  }
]
```

### AutoTune

Use the AutoTune function to tune BM25 parameters w.r.t. your document collection and queries.
All metrics supported by [ranx](https://github.com/AmenRa/ranx) are supported by the `autotune` function.

```python
best_params = se.autotune(
    queries=[{ "q_id": "q_1", "text": "...", ... }],  # Train queries
    qrels=[{ "q_1": { "doc_1": 1, ... }, ... }],      # Train qrels
    metric="ndcg@100",  # Default value, metric to maximize
    n_trials=100,       # Default value, number of trials
    n_res=100,          # Default value, number of results
)
```
Search using the best parameter configuration:
```
results = se.search(query, **best_params)
```


## 🎁 Feature Requests
Would you like to see other features implemented? Please, open a [feature request](https://github.com/AmenRa/retriv/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature+Request%5D+title).


## 🤘 Want to contribute?
Would you like to contribute? Please, drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20retriv).


## 📄 License
[retriv](https://github.com/AmenRa/retriv) is an open-sourced software licensed under the [MIT license](LICENSE).
