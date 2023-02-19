# Sparse Retriever

The Hybrid Retriever is searcher based on both lexical and semantic matching.
It comprises three components: the [Sparse Retriever]((https://github.com/AmenRa/retriv/blob/main/docs/sparse_retriever.md)), the [Dense Retriever]((https://github.com/AmenRa/retriv/blob/main/docs/dense_retriever.md)), and the Merger.
The Merger fuses the results of the Sparse and Dense Retrievers to compute the _hybrid_ results.

In the following, we show how to build an hybrid search engine, index a document collection, and search it.

## Build

You can instantiate and Hybrid Retriever as follows.
Default parameter values are shown.

```python
from retriv import HybridRetriever

sr = HybridRetriever(
    # Shared params ------------------------------------------------------------
    index_name="new-index",
    # Sparse retriever params --------------------------------------------------
    sr_model="bm25",
    min_df=1,
    tokenizer="whitespace",
    stemmer="english",
    stopwords="english",
    do_lowercasing=True,
    do_ampersand_normalization=True,
    do_special_chars_normalization=True,
    do_acronyms_normalization=True,
    do_punctuation_removal=True,
    # Dense retriever params ---------------------------------------------------
    dr_model="sentence-transformers/all-MiniLM-L6-v2",
    normalize=True,
    max_length=128,
    use_ann=True,
)
```

- Shared params:
  - `index_name`: [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index.
- Sparse Retriever params:
  - `sr_model`: defines the retrieval model to use for sparse retrieval (`bm25` or `tf-idf`).
  - `min_df`: terms that appear in less than `min_df` documents will be ignored.
  If integer, the parameter indicates the absolute count.
  If float, it represents a proportion of documents.
  - `tokenizer`: [tokenizer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable tokenizer or disable tokenization setting the parameter to `None`.
  - `stemmer`: [stemmer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable stemmer or disable stemming setting the parameter to `None`.
  - `stopwords`: [stopwords](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to remove during preprocessing. You can pass a custom stop-word list or disable stop-words removal by setting the parameter to `None`.
  - `do_lowercasing`: whether to lower case texts.
  - `do_ampersand_normalization`: whether to convert `&` in `and` during pre-processing.
  - `do_special_chars_normalization`: whether to remove special characters for letters, e.g., `übermensch` → `ubermensch`.
  - `do_acronyms_normalization`: whether to remove full stop symbols from acronyms without splitting them in multiple words, e.g., `P.C.I.` → `PCI`.
  - `do_punctuation_removal`: whether to remove punctuation.
- Dense Retriever params:
  - `dr_model`: defines the encoder model to encode queries and documents into vectors. You can use an [HuggingFace's Transformers](https://huggingface.co/models) pre-trained model by providing its ID or load a local model by providing its path.
  In the case of local models, the path must point to the directory containing the data saved with the [`PreTrainedModel.save_pretrained`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) method.
  Note that the representations are computed with `mean pooling` over the `last_hidden_state`.
  - `normalize`: whether to L2 normalize the vector representations.
  - `max_length`: texts longer than `max_length` will be automatically truncated. Choose this parameter based on how the employed model was trained or is generally used.
  - `use_ann`: whether to use approximate nearest neighbors search. Set it to `False` to use nearest neighbors search without approximation. If you have less than 20k documents in your collection, you probably want to disable approximation.

__Note:__ text pre-processing is equally applied to documents during indexing and to queries at search time.

## Index

### Create
You can index a document collection from JSONl, CSV, or TSV files.
CSV and TSV files must have a header.
[retriv](https://github.com/AmenRa/retriv) automatically infers the file kind, so there's no need to specify it.
Use the `callback` parameter to pass a function for converting your documents in the format supported by [retriv](https://github.com/AmenRa/retriv) on the fly.
Documents must have a single `text` field and an `id`.
The Hybrid Retriever sequentially build the indices for the Sparse and Dense Retrievers.
Indexes are automatically persisted on disk at the end of the process.
To speed up the indexing process of the Dense Retriever, you can activate GPU-based encoding.

```python
hr = hr.index_file(
  path="path/to/collection",  # File kind is automatically inferred
  embeddings_path=None,       # Default value
  use_gpu=False,              # Default value
  batch_size=512,             # Default value
  show_progress=True,         # Default value
  callback=lambda doc: {      # Callback defaults to None.
    "id": doc["id"],
    "text": doc["title"] + ". " + doc["text"],          
  ),
)
```

### Load
```python
hr = HybridRetriever.load("index-name")
```

### Delete
```python
HybridRetriever.delete("index-name")
```

## Search

During search, the Hybrid Retriever fuses the top 1000 results of the Sparse and Dense Retrievers.

### Search

Standard search functionality.

```python
hr.search(
  query="witches masses",    # What to search for        
  return_docs=True,          # Default value, return the text of the documents
  cutoff=100,                # Default value, number of results to return
)
```
Output:
```python
[
  {
    "id": "doc_2",
    "text": "Just like witches at black masses",
    "score": 0.9536403
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
hr.msearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,     # Default value, number of results
  batch_size=32,  # Default value.
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

- `batch_size`: how many searches to perform at once. Regulate it if you ran into memory usage issues or want to maximize throughput.

### Batch-Search

Batch-Search is similar to Multi-Search but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format.
[bsearch](#batch-search) is handy for computing results for hundreds of thousands or even millions of queries without hogging your RAM.

```python
hr.bsearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,
  batch_size=32,
  show_progress=True,
  qrels=None,   # To add relevance information to the saved files 
  path=None,    # Where to save the results, if you want to save them
)
```

## AutoTune

Use the AutoTune function to tune the Sparse Retriever's model [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) parameters and the importance given to the lexical and semantic relevance scores computed by the Sparse and Dense Retrievers, respectively.
All metrics supported by [ranx](https://github.com/AmenRa/ranx) are supported by the `autotune` function.
At the of the process, the best parameter configurations are automatically applied and saved to disk.
You can inspect the best configurations found by printing `hr.sparse_retriever.hyperparams`, `hr.merger.norm` and `hr.merger.params`.

```python
sr.autotune(
  queries=[{ "q_id": "q_1", "text": "...", ... }],  # Train queries
  qrels=[{ "q_1": { "doc_1": 1, ... }, ... }],      # Train qrels
  metric="ndcg",  # Default value, metric to maximize
  n_trials=100,   # Default value, number of trials
  cutoff=100,     # Default value, number of results
  batch_size=32,  # Default value
)
```