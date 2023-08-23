import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Union

import numba as nb
import numpy as np
import orjson
from numba.typed import List as TypedList
from oneliner_utils import create_path, read_jsonl
from tqdm import tqdm

from ..autotune import tune_bm25
from ..base_retriever import BaseRetriever
from ..paths import docs_path, fr_state_path
from ..sparse_retriever.preprocessing import (
    get_stemmer,
    get_stopwords,
    get_tokenizer,
    preprocessing,
    preprocessing_multi,
)
from ..sparse_retriever.sparse_retrieval_models.bm25 import bm25, bm25_multi
from ..sparse_retriever.sparse_retrieval_models.tf_idf import tf_idf, tf_idf_multi
from ..sparse_retriever.sparse_retriever import SparseRetriever, build_inverted_index
from ..utils.numba_utils import diff_sorted, intersect_sorted_multi, union_sorted_multi

CLAUSE_LIST = ["must", "must not"]
OPERATOR_LIST = ["eq", "gt", "gte", "lt", "lte", "between", "and", "or"]
KIND_LIST = ["id", "text", "number", "bool", "keyword", "keywords"]


class AdvancedRetriever(SparseRetriever):
    def __init__(
        self,
        schema: Dict[str, str],
        index_name: str = "new-index",
        model: str = "bm25",
        min_df: int = 1,
        tokenizer: Union[str, callable] = "whitespace",
        stemmer: Union[str, callable] = "english",
        stopwords: Union[str, List[str], Set[str]] = "english",
        do_lowercasing: bool = True,
        do_ampersand_normalization: bool = True,
        do_special_chars_normalization: bool = True,
        do_acronyms_normalization: bool = True,
        do_punctuation_removal: bool = True,
        hyperparams: dict = None,
    ):
        assert model.lower() in {"bm25", "tf-idf"}
        assert min_df > 0, "`min_df` must be greater than zero."
        self.check_schema(schema)
        self.init_args = {
            "schema": schema,
            "model": model.lower(),
            "min_df": min_df,
            "index_name": index_name,
            "do_lowercasing": do_lowercasing,
            "do_ampersand_normalization": do_ampersand_normalization,
            "do_special_chars_normalization": do_special_chars_normalization,
            "do_acronyms_normalization": do_acronyms_normalization,
            "do_punctuation_removal": do_punctuation_removal,
            "tokenizer": tokenizer,
            "stemmer": stemmer,
            "stopwords": stopwords,
        }

        self.schema = schema
        self.text_field = [k for k, v in self.schema.items() if v == "text"][0]
        self.model = model.lower()
        self.min_df = min_df
        self.index_name = index_name
        self.do_lowercasing = do_lowercasing
        self.do_ampersand_normalization = do_ampersand_normalization
        self.do_special_chars_normalization = do_special_chars_normalization
        self.do_acronyms_normalization = do_acronyms_normalization
        self.do_punctuation_removal = do_punctuation_removal
        self.tokenizer = get_tokenizer(tokenizer)
        self.stemmer = get_stemmer(stemmer)
        self.stopwords = [self.stemmer(sw) for sw in get_stopwords(stopwords)]
        self.id_mapping = None
        self.reversed_id_mapping = None
        self.inverted_index = None
        self.vocabulary = None
        self.doc_count = None
        self.doc_ids = None
        self.doc_lens = None
        self.avg_doc_len = None
        self.relative_doc_lens = None
        self.doc_index = None
        # self.mappings = None
        self.metadata = None

        self.preprocessing_kwargs = {
            "tokenizer": self.tokenizer,
            "stemmer": self.stemmer,
            "stopwords": self.stopwords,
            "do_lowercasing": self.do_lowercasing,
            "do_ampersand_normalization": self.do_ampersand_normalization,
            "do_special_chars_normalization": self.do_special_chars_normalization,
            "do_acronyms_normalization": self.do_acronyms_normalization,
            "do_punctuation_removal": self.do_punctuation_removal,
        }

        self.preprocessing_pipe = preprocessing_multi(**self.preprocessing_kwargs)

        self.hyperparams = dict(b=0.75, k1=1.2) if hyperparams is None else hyperparams

    def save(self) -> None:
        """Save the state of the retriever to be able to restore it later."""

        state = {
            "init_args": self.init_args,
            "id_mapping": self.id_mapping,
            "doc_count": self.doc_count,
            "inverted_index": self.inverted_index,
            "vocabulary": self.vocabulary,
            "doc_lens": self.doc_lens,
            "relative_doc_lens": self.relative_doc_lens,
            "hyperparams": self.hyperparams,
        }

        np.savez_compressed(fr_state_path(self.index_name), state=state)

    @staticmethod
    def load(index_name: str = "new-index"):
        """Load a retriever and its index.

        Args:
            index_name (str, optional): Name of the index. Defaults to "new-index".

        Returns:
            SparseRetriever: Sparse Retriever.
        """

        state = np.load(fr_state_path(index_name), allow_pickle=True)["state"][()]

        se = AdvancedRetriever(**state["init_args"])
        se.initialize_doc_index()
        se.id_mapping = state["id_mapping"]
        se.reversed_id_mapping = {v: k for k, v in state["id_mapping"].items()}
        se.doc_count = state["doc_count"]
        se.doc_ids = np.arange(state["doc_count"], dtype=np.int32)
        se.inverted_index = state["inverted_index"]
        se.vocabulary = set(se.inverted_index)
        se.doc_lens = state["doc_lens"]
        se.relative_doc_lens = state["relative_doc_lens"]
        se.hyperparams = state["hyperparams"]

        state = {
            "init_args": se.init_args,
            "id_mapping": se.id_mapping,
            "doc_count": se.doc_count,
            "inverted_index": se.inverted_index,
            "vocabulary": se.vocabulary,
            "doc_lens": se.doc_lens,
            "relative_doc_lens": se.relative_doc_lens,
            "hyperparams": se.hyperparams,
        }

        return se

    def check_schema(self, schema: Dict[str, str]) -> None:
        """Check if schema is valid"""
        text_found = False

        if "id" not in schema:
            raise ValueError("Schema must contain an id field")

        for k in schema:
            if not isinstance(k, str):
                raise TypeError("Schema keys must be strings")

        for value in schema.values():
            if value not in KIND_LIST:
                raise ValueError(f"Type {value} not supported")
            if value == "text":
                if text_found:
                    raise ValueError("Only one field can be text")
                text_found = True

        return True

    def check_collection(self, collection: Iterable, schema: Dict[str, str]) -> None:
        """Check collection against a schema"""
        for i, doc in enumerate(collection):
            if "id" not in doc:
                raise ValueError(f"Doc #{i} has no id")

            doc_id = doc["id"]

            for field in schema:
                if field not in doc:
                    raise ValueError(f"Field {field} not in doc {doc_id}")

            for field in doc:
                if field not in schema:
                    raise ValueError(f"Field {field} not in schema")

                kind = schema[field]
                value = doc[field]

                if kind == "id" and not isinstance(value, (int, str)):
                    raise TypeError(f"Field {field} of doc #{i} has wrong type")

                elif kind == "text" and not isinstance(value, str):
                    raise TypeError(f"Field {field} of doc {doc_id} has wrong type")

                elif kind == "number" and not isinstance(value, (int, float)):
                    raise TypeError(f"Field {field} of doc {doc_id} has wrong type")

                elif kind == "bool" and not isinstance(value, bool):
                    raise TypeError(f"Field {field} of doc {doc_id} has wrong type")

                elif kind == "keyword" and not isinstance(value, str):
                    raise TypeError(f"Field {field} of doc {doc_id} has wrong type")

                elif kind == "keywords" and not isinstance(value, (list, set, tuple)):
                    raise TypeError(f"Field {field} of doc {doc_id} has wrong type")

        return True

    def initialize_metadata(self, schema):
        metadata = {}

        for field, kind in schema.items():
            if kind == "number":
                metadata[field] = []
            if kind == "bool":
                metadata[field] = {True: [], False: []}
            elif kind in {"keyword", "keywords"}:
                metadata[field] = defaultdict(list)

        return metadata

    def fill_metadata(self, metadata, collection, schema):
        for i, doc in enumerate(collection):
            for field, kind in schema.items():
                if kind == "number":
                    metadata[field].append(doc[field])
                elif kind in ["bool", "keyword"]:
                    metadata[field][doc[field]].append(i)
                elif kind == "keywords":
                    for keyword in doc[field]:
                        metadata[field][keyword].append(i)

        return metadata

    def index_metadata(self, collection, schema):
        metadata = self.initialize_metadata(schema)
        metadata = self.fill_metadata(metadata, collection, schema)

        # Convert to numpy arrays
        for field, kind in schema.items():
            if kind == "number":
                metadata[field] = np.array(metadata[field])
            elif kind == "bool":
                metadata[field][True] = np.array(metadata[field][True], dtype=np.int32)
                metadata[field][False] = np.array(
                    metadata[field][False], dtype=np.int32
                )
            elif kind in ["keyword", "keywords"]:
                metadata[field] = dict(metadata[field])
                for keyword in metadata[field]:
                    metadata[field][keyword] = np.array(
                        metadata[field][keyword], dtype=np.int32
                    )
        return metadata

    def index_aux(self, text_field: str, show_progress: bool = True):
        """Internal usage."""
        collection = read_jsonl(
            docs_path(self.index_name),
            generator=True,
            callback=lambda x: x[text_field],
        )

        # Preprocessing --------------------------------------------------------
        collection = self.preprocessing_pipe(collection, generator=True)

        # Inverted index -------------------------------------------------------
        (
            self.inverted_index,
            self.doc_lens,
            self.relative_doc_lens,
        ) = build_inverted_index(
            collection=collection,
            n_docs=self.doc_count,
            min_df=self.min_df,
            show_progress=show_progress,
        )
        self.avg_doc_len = np.mean(self.doc_lens, dtype=np.float32)
        self.vocabulary = set(self.inverted_index)

    def index(
        self,
        collection: Iterable,
        callback: callable = None,
        show_progress: bool = True,
    ):
        """Index a given collection of documents.

        Args:
            collection (Iterable): collection of documents to index.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            SparseRetriever: Sparse Retriever.
        """
        self.check_collection(collection, self.schema)
        self.save_collection(collection, callback)
        self.initialize_doc_index()
        self.initialize_id_mapping()
        self.reversed_id_mapping = {v: k for k, v in self.id_mapping.items()}
        self.doc_count = len(self.id_mapping)
        self.doc_ids = np.arange(self.doc_count, dtype=np.int32)
        self.index_aux(
            text_field=self.text_field,
            show_progress=show_progress,
        )
        self.metadata = self.index_metadata(collection, schema=self.schema)
        self.save()
        return self

    def index_file(
        self, path: str, callback: callable = None, show_progress: bool = True
    ):
        """Index the collection contained in a given file.

        Args:
            path (str): path of file containing the collection to index.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            SparseRetriever: Sparse Retriever
        """

        # collection = self.collection_generator(path=path, callback=callback)
        collection_generator = self.collection_generator

        class Collection:
            def __init__(self, path, callback):
                self.path = path
                self.callback = callback

            def __iter__(self):
                yield from collection_generator(path=self.path, callback=self.callback)

        return self.index(
            collection=Collection(path, callback), show_progress=show_progress
        )

    def filter_doc_ids(
        self,
        field: str,
        clause: str,
        value: Any = None,
        operator: str = None,
        raise_error: bool = True,
    ):
        if clause not in CLAUSE_LIST:
            raise ValueError(f"Clause must be one of {CLAUSE_LIST}")
        if operator is not None and operator not in OPERATOR_LIST:
            raise ValueError(f"Operator must be one of {OPERATOR_LIST}")
        if field not in self.schema:
            raise ValueError(f"Field `{field}` not in schema")

        kind = self.schema[field]
        doc_ids = self.doc_ids
        id_mapping = self.reversed_id_mapping
        metadata = self.metadata

        def get_value(field, value):
            if raise_error and value not in metadata[field]:
                raise ValueError(f"No document has value `{value}` in field `{field}`.")

            return metadata[field].get(value, np.array([], dtype=np.int32))

        if kind == "id":
            if clause == "must":
                return doc_ids[np.isin(doc_ids, [id_mapping[v] for v in value])]
            elif clause == "must not":
                return doc_ids[~np.isin(doc_ids, [id_mapping[v] for v in value])]

        elif kind == "bool":
            if clause == "must":
                return metadata[field][value]
            elif clause == "must not":
                return metadata[field][not value]

        elif kind == "keyword":
            if clause == "must":
                if isinstance(value, list):
                    return union_sorted_multi(
                        TypedList([get_value(field, v) for v in value])
                    )
                else:
                    return get_value(field, value)

            elif clause == "must not":
                if isinstance(value, list):
                    ids = [
                        get_value(field, v) for v in metadata[field] if v not in value
                    ]

                else:
                    ids = [get_value(field, v) for v in metadata[field] if v != value]

                return union_sorted_multi(TypedList(ids))

        elif kind == "keywords":
            if clause == "must":
                if isinstance(value, list):
                    if operator == "and":
                        return intersect_sorted_multi(
                            TypedList([get_value(field, v) for v in value])
                        )
                    elif operator == "or":
                        return union_sorted_multi(
                            TypedList([get_value(field, v) for v in value])
                        )
                    else:
                        raise ValueError(
                            f"Operator `{operator}`not supported for keywords field"
                        )
                else:
                    return get_value(field, value)

            elif clause == "must not":
                if isinstance(value, list):
                    must_not_ids = [get_value(field, v) for v in value]
                    if operator == "and":
                        must_not_ids = intersect_sorted_multi(TypedList(must_not_ids))
                    elif operator == "or":
                        must_not_ids = union_sorted_multi(TypedList(must_not_ids))
                    else:
                        raise ValueError(
                            f"Operator `{operator}`not supported for keywords field"
                        )

                    return diff_sorted(
                        np.arange(self.doc_count, dtype=np.int32), must_not_ids
                    )

                else:
                    return diff_sorted(
                        np.arange(self.doc_count, dtype=np.int32),
                        get_value(field, value),
                    )

        elif kind == "number":
            if operator == "eq":
                mask = metadata[field] == value
            elif operator == "gt":
                mask = metadata[field] > value
            elif operator == "gte":
                mask = metadata[field] >= value
            elif operator == "lt":
                mask = metadata[field] < value
            elif operator == "lte":
                mask = metadata[field] <= value
            elif operator == "between":
                data, min_v, max_v = metadata[field], value[0], value[1]
                mask = np.logical_and(data >= min_v, data <= max_v)
            else:
                raise ValueError("Operator not supported for numeric field")

            if clause == "must":
                return doc_ids[mask]
            elif clause == "must not":
                return doc_ids[~mask]

        else:
            raise ValueError(
                f"Field {field} of type {kind} not supported for filtering"
            )

    def get_filtered_doc_ids(self, filters: List[Dict]) -> np.ndarray:
        if len(filters) == 1:
            return self.filter_doc_ids(**filters[0])
        filtered_doc_ids = TypedList([self.filter_doc_ids(**f) for f in filters])
        return intersect_sorted_multi(filtered_doc_ids)

    def format_filters(self, filters: Dict, clause: str = "must") -> List[Dict]:
        formatted_filters = []

        for field, value in filters.items():
            if self.schema[field] in {"id", "bool", "keyword"}:
                f = dict(field=field, clause=clause, value=value)

            elif self.schema[field] in {"number", "keywords"}:
                f = dict(field=field, clause=clause, value=value[1], operator=value[0])

            formatted_filters.append(f)

        return formatted_filters

    def search(
        self,
        query: Union[Dict, str],
        return_docs: bool = True,
        cutoff: int = 100,
        operator: str = "OR",
        subset_doc_ids: List = None,
    ) -> List:
        if isinstance(query, str):
            query_text = query
            if subset_doc_ids is not None:
                subset_doc_ids = np.array(
                    [self.reversed_id_mapping[doc_id] for doc_id in subset_doc_ids],
                    dtype=np.int32,
                )
        else:
            query_text = query.get("text", "")
            must_filters = query.get("where", {})
            must_not_filters = query.get("where_not", {})
            must_single_filters = {
                k: v
                for k, v in query.items()
                if k not in {"text", "where", "where_not"}
            }

            must_filters = self.format_filters(must_filters)
            must_not_filters = self.format_filters(must_not_filters, clause="must not")
            must_single_filters = self.format_filters(must_single_filters)
            filters = must_filters + must_not_filters + must_single_filters
            subset_doc_ids = self.get_filtered_doc_ids(filters)

        query_terms = self.query_preprocessing(query_text)
        query_terms = [t for t in query_terms if t in self.vocabulary]

        if query_terms:
            doc_ids = self.get_doc_ids(query_terms)
            term_doc_freqs = self.get_term_doc_freqs(query_terms)

            if self.model == "bm25":
                unique_doc_ids, scores = bm25(
                    term_doc_freqs=term_doc_freqs,
                    doc_ids=doc_ids,
                    relative_doc_lens=self.relative_doc_lens,
                    doc_count=self.doc_count,
                    cutoff=cutoff,
                    operator=operator,
                    subset_doc_ids=subset_doc_ids,
                    **self.hyperparams,
                )
            elif self.model == "tf-idf":
                unique_doc_ids, scores = tf_idf(
                    term_doc_freqs=term_doc_freqs,
                    doc_ids=doc_ids,
                    doc_lens=self.doc_lens,
                    cutoff=cutoff,
                    operator=operator,
                    subset_doc_ids=subset_doc_ids,
                )
            else:
                raise NotImplementedError()
        else:
            if subset_doc_ids is None:
                unique_doc_ids = self.doc_ids
                scores = np.ones(self.doc_count)
            else:
                unique_doc_ids = subset_doc_ids
                scores = np.ones(len(subset_doc_ids))

        unique_doc_ids = self.map_internal_ids_to_original_ids(unique_doc_ids)

        if not return_docs:
            return dict(zip(unique_doc_ids, scores))

        return self.prepare_results(unique_doc_ids, scores)
