# Sparse Retriever

The Sparse Retriever is a traditional searcher based on lexical matching.
It supports [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), the retrieval model used by major search engines libraries, such as [Lucene](https://en.wikipedia.org/wiki/Apache_Lucene) and [Elasticsearch](https://en.wikipedia.org/wiki/Elasticsearch). 
[retriv](https://github.com/AmenRa/retriv) also implements the classic relevance model [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) for educational purposes.

The Sparse Retriever also provides several resources for multi-lingual [text pre-processing](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md), aiming to maximize its retrieval effectiveness.

In the following, we show how to build a search engine employing a sparse retriever, index a document collection, and search it.

## Build

The Sparse Retriever provides several options to tailor its functioning to you preferences, as shown below.
Default parameter values are shown.

```python
# Note: the SparseRetriever has an alias called SearchEngine, if you prefer
from retriv import SparseRetriever

sr = SparseRetriever(
  index_name="new-index",
  model="bm25",
  min_df=1,
  tokenizer="whitespace",
  stemmer="english",
  stopwords="english",
  do_lowercasing=True,
  do_ampersand_normalization=True,
  do_special_chars_normalization=True,
  do_acronyms_normalization=True,
  do_punctuation_removal=True,
)
```

- `index_name`: [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index.
- `model`: defines the retrieval model to use for searching (`bm25` or `tf-idf`).
- `min_df`: terms that appear in less than `min_df` documents will be ignored.
If integer, the parameter indicates the absolute count.
If float, it represents a proportion of documents.
- `tokenizer`: [tokenizer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable tokenizer or disable tokenization by setting the parameter to `None`.
- `stemmer`: [stemmer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable stemmer or disable stemming setting the parameter to `None`.
- `stopwords`: [stopwords](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to remove during preprocessing. You can pass a custom stop-word list or disable stop-words removal by setting the parameter to `None`.
- `do_lowercasing`: whether to lowercase texts.
- `do_ampersand_normalization`: whether to convert `&` in `and` during pre-processing.
- `do_special_chars_normalization`: whether to remove special characters for letters, e.g., `übermensch` → `ubermensch`.
- `do_acronyms_normalization`: whether to remove full stop symbols from acronyms without splitting them in multiple words, e.g., `P.C.I.` → `PCI`.
- `do_punctuation_removal`: whether to remove punctuation.

__Note:__ text pre-processing is equally applied to documents during indexing and to queries at search time.

## Index

### Create
You can index a document collection from JSONl, CSV, or TSV files.
CSV and TSV files must have a header.
[retriv](https://github.com/AmenRa/retriv) automatically infers the file kind, so there's no need to specify it.
Use the `callback` parameter to pass a function for converting your documents in the format supported by [retriv](https://github.com/AmenRa/retriv) on the fly.
Documents must have a single `text` field and an `id`.
Indexes are automatically persisted on disk at the end of the process.
Indexing functionalities are built to have minimal memory footprint while leveraging multi-processing for efficiency.
Indexing 10M documents takes from 5 to 10 minutes on a [AMD Ryzen™ 9 5950X](https://www.amd.com/en/products/cpu/amd-ryzen-9-5950x), depending on the length of the documents.

```python
sr = sr.index_file(
  path="path/to/collection",  # File kind is automatically inferred
  show_progress=True,         # Default value
  callback=lambda doc: {      # Callback defaults to None.
    "id": doc["id"],
    "text": doc["title"] + ". " + doc["text"],          
  )
```

### Load
```python
sr = SparseRetriever.load("index-name")
```

### Delete
```python
SparseRetriever.delete("index-name")
```

## Search

### Search

Standard search functionality.

```python
sr.search(
  query="witches masses",    # What to search for        
  return_docs=True,          # Default value, return the text of the documents
  cutoff=100,                # Default value, number of results to return
)
```
Output:
```json
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

Compute results for multiple queries at once.

```python
sr.msearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,  # Default value, number of results
)
```
Output:
```json
{
  "q_1": {
    "doc_2": 1.7536403,
    "doc_1": 0.6931472
  },
  ...
}
```

### Batch-Search

Batch-Search is similar to Multi-Search but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format.
[bsearch](#batch-search) is handy for computing results for hundreds of thousands or even millions of queries without hogging your RAM.

```python
sr.bsearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,
  batch_size=1_000,
  show_progress=True,
  qrels=None,   # To add relevance information to the saved files 
  path=None,    # Where to save the results, if you want to save them
)
```

## AutoTune

Use the AutoTune function to tune [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) parameters w.r.t. your document collection and queries.
All metrics supported by [ranx](https://github.com/AmenRa/ranx) are supported by the `autotune` function.
At the of the process, the best parameter configuration is automatically applied to the `SparseRetriever` instance and saved to disk.
You can inspect the current configuration by printing `sr.hyperparams`.

```python
sr.autotune(
  queries=[{ "q_id": "q_1", "text": "...", ... }],  # Train queries
  qrels={ "q_1": { "doc_1": 1, ... }, ... },      # Train qrels
  metric="ndcg",  # Default value, metric to maximize
  n_trials=100,   # Default value, number of trials
  cutoff=100,     # Default value, number of results
)
```
