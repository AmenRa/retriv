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

[retriv](https://github.com/AmenRa/retriv) is a fast [search engine](https://en.wikipedia.org/wiki/Search_engine) implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), leveraging [Numba](https://github.com/numba/numba) for high-speed [vector operations](https://en.wikipedia.org/wiki/Automatic_vectorization) and [automatic parallelization](https://en.wikipedia.org/wiki/Automatic_parallelization).
It offers a user-friendly interface to index and search your document collection and allows you to automatically tune the underling retrieval model, [BM25](https://en.wikipedia.org/wiki/Okapi_BM25).

[How fast is your retriv?](#speed-comparison)


## ✨ Features

### Retrieval Models
[retriv](https://github.com/AmenRa/retriv) implements [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) as a retrieval model. Alternatives will probably be added in the future.


### Multi-search & Batch-search
In addition to the standard [search](#search) functionality, [retriv](https://github.com/AmenRa/retriv) provides two additional search methods: [msearch](#multi-search) and [bsearch](#batch-search).
- [msearch](#multi-search) allows computing the results for multiple queries at once, leveraging the [automatic parallelization](https://en.wikipedia.org/wiki/Automatic_parallelization) features offered by [Numba](https://github.com/numba/numba).
- [bsearch](#batch-search) is similar to [msearch](#multi-search) but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format. [bsearch](#batch-search) is very useful for pre-computing [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) results for hundred of thousands or even millions of queries without hogging your RAM. Pre-computed results can be leveraged for negative sampling during the training of [Neural Models](https://en.wikipedia.org/wiki/Artificial_neural_network) for [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval).


### AutoTune
[retriv](https://github.com/AmenRa/retriv) offers an automatic tuning functionality that allows you to tune [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)'s parameters with a single function call.
Under the hood, [retriv](https://github.com/AmenRa/retriv) leverages [Optuna](https://optuna.org), a [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) framework, and [ranx](https://github.com/AmenRa/ranx), an [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval) evaluation library, to test several parameter configurations for [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and choose the best one.


### Stemmers
[Stemmers](https://en.wikipedia.org/wiki/Stemming) reduce words to their word stem, base or root form.  
[retriv](https://github.com/AmenRa/retriv) supports the following stemmers:
- [snowball](https://snowballstem.org) (default)  
The following languages are supported by Snowball Stemmer: 
Arabic, Basque, Catalan, Danish, Dutch, English, Finnish, French, German, Greek, Hindi, Hungarian, Indonesian, Irish, Italian, Lithuanian, Nepali, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish.  
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


### Automatic Spell Correction
[retriv](https://github.com/AmenRa/retriv) provides automatic spell correction through [Hunspell](http://hunspell.github.io) for [92 languages](https://github.com/wooorm/dictionaries#list-of-dictionaries).
Please, follow the link and choose your preferred language (e.g., Italian → "dictionary-it" → use "it").
For some languages you can directly pass their names: Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Portuguese, Romanian, Russian, Spanish, and Swedish.  

_NOTE: Automatic spell correction is disabled by default. It can introduce artifacts, degrading retrieval performances when documents are free from misspellings. If possible, check whether it can improve retrieval performances for your specific document collection._

## 🔌 Installation
```bash
pip install retriv
```



## 💡 Usage

### Minimal Working Example

```python
from retriv import SearchEngine

collection = [
  {"id": "doc_1", "text": "Generals gathered in their masses"},
  {"id": "doc_2", "text": "Just like witches at black masses"},
  {"id": "doc_3", "text": "Evil minds that plot destruction"},
  {"id": "doc_4", "text": "Sorcerer of death's construction"},
]

se = SearchEngine("new-index")
se.index(collection)

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

### Create index from file
You can index a document collection from a JSONl, CSV, or TSV file.
CSV and TSV files must have a header.
File kind is automatically inferred.
Use the `callback` parameter to pass a function for converting your documents in the format supported by [retriv](https://github.com/AmenRa/retriv) on the fly.
Indexes are automatically saved.
This is the preferred way of creating indexes as it has a low memory footprint.

```python
from retriv import SearchEngine

se = SearchEngine("new-index")

se.index_file(
  path="path/to/collection",  # File kind is automatically inferred
  show_progress=True,         # Default value
  callback=lambda doc: {      # Callback defaults to None
    "id": doc["id"],
    "text": doc["title"] + "\n" + doc["body"],          
  )
```

`se = SearchEngine("new-index")` is equivalent to:
```python
se = SearchEngine(
  index_name="new-index",               # Default value
  min_df=1,                             # Min doc-frequency. Defaults to 1.
  tokenizer="whitespace",               # Default value
  stemmer="english",                    # Default value (Snowball English)
  stopwords="english",                  # Default value
  spell_corrector=None,                 # Default value
  do_lowercasing=True,                  # Default value
  do_ampersand_normalization=True,      # Default value
  do_special_chars_normalization=True,  # Default value
  do_acronyms_normalization=True,       # Default value
  do_punctuation_removal=True,          # Default value
)
```


### Create index from list 
```python
collection = [
  {"id": "doc_1", "title": "...", "body": "..."},
  {"id": "doc_2", "title": "...", "body": "..."},
  {"id": "doc_3", "title": "...", "body": "..."},
  {"id": "doc_4", "title": "...", "body": "..."},
]

se = SearchEngine(...)

se.index(
  collection,
  show_progress=True,         # Default value
  callback=lambda doc: {      # Callback defaults to None
    "id": doc["id"],
    "text": doc["title"] + "\n" + doc["body"],          
  )
)
```


### Load / Delete index
```python
from retriv import SearchEngine

se = SearchEngine.load("index-name")

SearchEngine.delete("index-name")
```


### Search
```python
se.search(
  query="witches masses",
  return_docs=True,  # Default value
  cutoff=100,        # Default value, number of results to return
)
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


### Multi-Search
```python
se.msearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,  # Default value, number of results
)
```
Output:
```python
{
  "q_1": {
    "doc_2": 1.7536403,
    "doc_1": 0.6931472
  },
  ...
}
```


### AutoTune

Use the AutoTune function to tune [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) parameters w.r.t. your document collection and queries.
All metrics supported by [ranx](https://github.com/AmenRa/ranx) are supported by the `autotune` function.

```python
se.autotune(
    queries=[{ "q_id": "q_1", "text": "...", ... }],  # Train queries
    qrels=[{ "q_1": { "doc_1": 1, ... }, ... }],      # Train qrels
    metric="ndcg",  # Default value, metric to maximize
    n_trials=100,   # Default value, number of trials
    cutoff=100,     # Default value, number of results
)
```

At the of the process, the best parameter configuration is automatically applied to the `SearchEngine` instance and saved to disk.
You can see what the configuration is by printing `se.hyperparams`.

## Speed Comparison

We performed a speed test, comparing [retriv](https://github.com/AmenRa/retriv) to [rank_bm25](https://github.com/dorianbrown/rank_bm25), a popular [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) implementation in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and [pyserini](https://github.com/castorini/pyserini), a [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) binding to the [Lucene](https://en.wikipedia.org/wiki/Apache_Lucene) search engine.

We relied on the [MSMARCO Passage](https://microsoft.github.io/msmarco) dataset to collect documents and queries.
Specifically, we used the original document collection and three sub-samples of it, accounting for 1k, 100k, and 1M documents, respectively, and sampled 1k queries from the original ones.
We computed the top-100 results with each library (if possible). 
Results are reported below. Best results are highlighted in boldface.

| Library                                                   | Collection Size | Elapsed Time | Avg. Query Time | Throughput (q/s) |
| --------------------------------------------------------- | --------------: | -----------: | --------------: | ---------------: |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |           1,000 |        646ms |           6.5ms |           1548/s |
| [pyserini](https://github.com/castorini/pyserini)         |           1,000 |      1,438ms |           1.4ms |            695/s |
| [retriv](https://github.com/AmenRa/retriv)                |           1,000 |        140ms |           0.1ms |           7143/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |           1,000 |    __134ms__ |       __0.1ms__ |       __7463/s__ |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |         100,000 |    106,000ms |          1060ms |              1/s |
| [pyserini](https://github.com/castorini/pyserini)         |         100,000 |      2,532ms |           2.5ms |            395/s |
| [retriv](https://github.com/AmenRa/retriv)                |         100,000 |        314ms |           0.3ms |           3185/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |         100,000 |    __256ms__ |       __0.3ms__ |       __3906__/s |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |       1,000,000 |          N/A |             N/A |              N/A |
| [pyserini](https://github.com/castorini/pyserini)         |       1,000,000 |      4,060ms |           4.1ms |            246/s |
| [retriv](https://github.com/AmenRa/retriv)                |       1,000,000 |      1,018ms |           1.0ms |            982/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |       1,000,000 |    __503ms__ |       __0.5ms__ |       __1988/s__ |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |       8,841,823 |          N/A |             N/A |              N/A |
| [pyserini](https://github.com/castorini/pyserini)         |       8,841,823 |     12,245ms |          12.2ms |             82/s |
| [retriv](https://github.com/AmenRa/retriv)                |       8,841,823 |     10,763ms |          10.8ms |             93/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |       8,841,823 |  __4,476ms__ |       __4.4ms__ |        __227/s__ |


## 🎁 Feature Requests
Would you like to see other features implemented? Please, open a [feature request](https://github.com/AmenRa/retriv/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature+Request%5D+title).


## 🤘 Want to contribute?
Would you like to contribute? Please, drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20retriv).


## 📄 License
[retriv](https://github.com/AmenRa/retriv) is an open-sourced software licensed under the [MIT license](LICENSE).