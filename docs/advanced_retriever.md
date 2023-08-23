# Advanced Retriever

⚠️ This is an experimental feature.

The Advanced Retriever is a searcher based on lexical matching and search filters.
It supports [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) as the [Sparse Retriever](https://github.com/AmenRa/retriv/blob/main/docs/sparse_retriever.md) and provides the same resources for multi-lingual [text pre-processing](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md). In addition, it supports search filters, i.e., a set of rules that can be used to filter out documents from the search results.

In the following, we show how to build a search engine employing an advanced retriever, index a document collection, and search it.

## Schema

The first step to building an Advanced Retriever is to define the `schema` of document collection.
The `schema` is a dictionary describing the documents' `fields` and their `data types`.
Based on the `data types`, search `filters` can be defined and applied to the search results.

[retriv](https://github.com/AmenRa/retriv) supports the following data types:
- __id:__ field used for the document IDs.
- __text:__ text field used for lexical matching.
- __number:__ numeric value.
- __bool:__ boolean value (True or False).
- __keyword:__ string or number representing a keyword or a category.
- __keywords:__ list of keywords.

An example of `schema` for a collection of books is shown below.  
NB: At the time of writing, [retriv](https://github.com/AmenRa/retriv) supports only one text field per schema.
Therefore, the `content` field is used for both the title and the abstract of the books.

```json
schema = {
  "isbn": "id",
  "content": "text",
  "year": "number",
  "is_english": "bool",
  "author": "keyword",
  "genres": "keywords",
}
```

## Build

The Advanced Retriever provides several options to tailor its functioning to you preferences, as shown below.

```python
from retriv.experimental import AdvancedRetriever

ar = AdvancedRetriever(
  schema=schema,
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

- `schema`: the documents' schema.
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
Use the `callback` parameter to pass a function for converting your documents in the format defined by your `schema` on the fly.
Indexes are automatically persisted on disk at the end of the process.

```python
ar = ar.index_file(
  path="path/to/collection",  # File kind is automatically inferred
  show_progress=True,         # Default value
  callback=lambda doc: {      # Callback defaults to None.
    "id": doc["id"],
    "text": doc["title"] + ". " + doc["text"],       
    ...   
  )
```

### Load
```python
ar = AdvancedRetriever.load("index-name")
```

### Delete
```python
AdvancedRetriever.delete("index-name")
```

## Search

### Query & Filters

Advanced Retriever search query can be either a string or a dictionary.
In the former case, the string is used as the query text and no filters are applied.
In the latter case, the dictionary defines the query text and the filters to apply to the search results. If the query text is omitted from the dictionary, documents matching the filters will be returned.

[retriv](https://github.com/AmenRa/retriv) supports two way of filtering the search results (`where` and `where_not`) and several type-specific operators.

- `where` means that only the documents matching the filter will be considered during search.
- `where_not` means that the documents matching the filter will be ignored during search.

Below we describe the effects of the supported operators for each data type and way of filtering.

#### Where

| Field Type | Operator  | Value                      | Meaning                                                                                                                                              |
| ---------- | --------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| number     | `eq`      | number                     | Only the documents whose field value is **equal to** the provided value will be considered during search.                                            |
| number     | `gt`      | number                     | Only the documents whose field value is **greater than** the provided value will be considered during search.                                        |
| number     | `gte`     | number                     | Only the documents whose field value is **greater or equal to** the provided value will be considered during search.                                 |
| number     | `lt`      | number                     | Only the documents whose field value is **less than** the provided value will be considered during search.                                           |
| number     | `lte`     | number                     | Only the documents whose field value is **less or equal to** the provided value will be considered during search.                                    |
| number     | `between` | number                     | Only the documents whose field value is **between** the provided values (included) will be considered during search.                                 |
| bool       |           | True / False               | Only the documents whose field value is **equal to** the provided value will be considered during search.                                            |
| keyword    |           | any value / list of values | Only the documents whose field value is **equal to** the provided value or **among** the provided values will be considered during search.           |
| keywords   | `or`      | any value / list of values | Only the documents whose field value is **contains** the provided value or **contains one of** the provided values will be considered during search. |
| keywords   | `and`     | any value / list of values | Only the documents whose field value **contains all** the provided values will be considered during search.                                          |

Query example:
```python
query = {
    "text": "search terms",
    "where": {
        "numeric_field_name": ("gte", 1970),
        "boolean_field_name": True,
        "keyword_field_name": "kw_1",
        "keywords_field_name": ("or", ["kws_23", "kws_666"]),
    }
}
```

Alternatively, you can omit the `where` key and use the following syntax:
```python
query = {
    "text": "search terms",
    "numeric_field_name": ("gte", 1970),
    "boolean_field_name": True,
    "keyword_field_name": "kw_1",
    "keywords_field_name": ("or", ["kws_23", "kws_666"]),
}
```


#### Where not

| Field Type | Operator  | Value                      | Meaning                                                                                                                        |
| ---------- | --------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| number     | `eq`      | number                     | The documents whose field value is **equal to** the provided value will be ignored.                                            |
| number     | `gt`      | number                     | The documents whose field value is **greater than** the provided value will be ignored.                                        |
| number     | `gte`     | number                     | The documents whose field value is **greater or equal to** the provided value will be ignored.                                 |
| number     | `lt`      | number                     | The documents whose field value is **less than** the provided value will be ignored.                                           |
| number     | `lte`     | number                     | The documents whose field value is **less or equal to** the provided value will be ignored.                                    |
| number     | `between` | number                     | The documents whose field value is **between** the provided values (included) will be ignored.                                 |
| bool       |           | True / False               | The documents whose field value is **equal to** the provided value will be ignored.                                            |
| keyword    |           | any value / list of values | The documents whose field value is **equal to** the provided value or **among** the provided values will be ignored.           |
| keywords   | `or`      | any value / list of values | The documents whose field value is **contains** the provided value or **contains one of** the provided values will be ignored. |
| keywords   | `and`     | any value / list of values | The documents whose field value **contains all** the provided values will be ignored.                                          |

Query example:
```python
query = {
    "text": "search terms",
    "where": {
        "numeric_field_name": ("gte", 1970),
        "boolean_field_name": True,
        "keyword_field_name": "kw_1",
        "keywords_field_name": ("or", ["kws_23", "kws_666"]),
    }
}
```

### Search

```python
ar.search(
  query: ... 
  return_docs=True     # Default value.
  cutoff=100           # Default value.
  operator="OR"        # Default value.
  subset_doc_ids=None  # Default value.
)
```

- `query`: what to search for and which filters to apply. See the section [Query & Filters](#query--filters) for more details.
- `return_docs`: whether to return documents or only their IDs.
- `cutoff`: number of results to return.
- `operator`: whether to perform conjunctive (`AND`) or disjunctive (`OR`) search.  Conjunctive search retrieves documents that contain **all** the query terms. Disjunctive search retrieves documents that contain **at least one** of the query terms.
- `subset_doc_ids`: restrict the search to the subset of documents having the provided IDs.

Sample output:
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




<!-- ### Multi-Search

Compute results for multiple queries at once.

```python
ar.msearch(
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
ar.bsearch(
  queries=[{"id": "q_1", "text": "witches masses"}, ...],
  cutoff=100,
  batch_size=1_000,
  show_progress=True,
  qrels=None,   # To add relevance information to the saved files 
  path=None,    # Where to save the results, if you want to save them
)
``` -->

<!-- ## AutoTune

Use the AutoTune function to tune [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) parameters w.r.t. your document collection and queries.
All metrics supported by [ranx](https://github.com/AmenRa/ranx) are supported by the `autotune` function.
At the of the process, the best parameter configuration is automatically applied to the `AdvancedRetriever` instance and saved to disk.
aou can inspect the current configuration by printing `ar.hyperparams`.

```python
ar.autotune(
  queries=[{ "q_id": "q_1", "text": "...", ... }],  # Train queries
  qrels=[{ "q_1": { "doc_1": 1, ... }, ... }],      # Train qrels
  metric="ndcg",  # Default value, metric to maximize
  n_trials=100,   # Default value, number of trials
  cutoff=100,     # Default value, number of results
)
``` -->
