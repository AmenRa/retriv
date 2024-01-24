# Dense Retriever

The Dense Retriever performs [semantic search](https://en.wikipedia.org/wiki/Semantic_search), i.e., it compares vector representations of queries and documents to compute the relevance scores of the latter.
The Dense Retriever comprises two components: the `Encoder` and the `ANN Searcher`, described below.

In the following, we show how to build a search engine for [semantic search](https://en.wikipedia.org/wiki/Semantic_search) based on dense retrieval, index a document collection, and search it.

## Build

Building a Dense Retriever is as simple as shown below.
Default parameter values are shown.

```python
from retriv import DenseRetriever

dr = DenseRetriever(
  index_name="new-index",
  model="sentence-transformers/all-MiniLM-L6-v2",
  normalize=True,
  max_length=128,
  use_ann=True,
)
```

- `index_name`: [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index.
- `model`: defines the encoder model to encode queries and documents into vectors. You can use an [HuggingFace's Transformers](https://huggingface.co/models) pre-trained model by providing its ID or load a local model by providing its path.
In the case of local models, the path must point to the directory containing the data saved with the [`PreTrainedModel.save_pretrained`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) method.
Note that the representations are computed with `mean pooling` over the `last_hidden_state`.
- `normalize`: whether to L2 normalize the vector representations.
- `max_length`: texts longer than `max_length` will be automatically truncated. Choose this parameter based on how the employed model was trained or is generally used.
- `use_ann`: whether to use approximate nearest neighbors search. Set it to `False` to use nearest neighbors search without approximation. If you have less than 20k documents in your collection, you probably want to disable approximation.

## Index

### Create
You can index a document collection from JSONl, CSV, or TSV files.
CSV and TSV files must have a header.
[retriv](https://github.com/AmenRa/retriv) automatically infers the file kind, so there's no need to specify it.
Use the `callback` parameter to pass a function for converting your documents in the format supported by [retriv](https://github.com/AmenRa/retriv) on the fly.
Documents must have a single `text` field and an `id`.
Indexes are automatically persisted on disk at the end of the process.
To speed up the indexing process, you can activate GPU-based encoding.

```python
dr = dr.index_file(
  path="path/to/collection",  # File kind is automatically inferred
  embeddings_path=None,       # Default value
  use_gpu=False,              # Default value
  batch_size=512,             # Default value
  show_progress=True,         # Default value
  callback=lambda doc: {      # Callback defaults to None.
    "id": doc["id"],
    "text": doc["title"] + ". " + doc["text"],          
  },
)
```

- `embeddings_path`: in case you want to load pre-computed embeddings, you can provide the path to a `.npy` file. Embeddings must be in the same order as the documents in the collection file.
- `use_gpu`: whether to use the GPU for document encoding.
- `batch_size`: how many documents to encode at once. Regulate it if you ran into memory usage issues or want to maximize throughput.


### Load
```python
dr = DenseRetriever.load("index-name")
```

### Delete
```python
DenseRetriever.delete("index-name")
```

## Search

### Search

Standard search functionality.

```python
dr.search(
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
dr.msearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,     # Default value, number of results
  batch_size=32,  # Default value.
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

- `batch_size`: how many searches to perform at once. Regulate it if you ran into memory usage issues or want to maximize throughput.

### Batch-Search

Batch-Search is similar to Multi-Search but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format.
[bsearch](#batch-search) is handy for computing results for hundreds of thousands or even millions of queries without hogging your RAM.

```python
dr.bsearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,
  batch_size=32,
  show_progress=True,
  qrels=None,   # To add relevance information to the saved files 
  path=None,    # Where to save the results, if you want to save them
)
```
