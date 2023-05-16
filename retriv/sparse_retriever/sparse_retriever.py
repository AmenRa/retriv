import os
from typing import Dict, Iterable, List, Set, Union

import numba as nb
import numpy as np
import orjson
from numba.typed import List as TypedList
from oneliner_utils import create_path, read_jsonl
from tqdm import tqdm

from ..autotune import tune_bm25
from ..base_retriever import BaseRetriever
from ..paths import docs_path, sr_state_path
from .build_inverted_index import build_inverted_index
from .preprocessing import (
    get_stemmer,
    get_stopwords,
    get_tokenizer,
    multi_preprocessing,
    preprocessing,
)
from .sparse_retrieval_models.bm25 import bm25, bm25_multi
from .sparse_retrieval_models.tf_idf import tf_idf, tf_idf_multi


class SparseRetriever(BaseRetriever):
    def __init__(
        self,
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
        """The Sparse Retriever is a traditional searcher based on lexical matching. It supports BM25, the retrieval model used by major search engines libraries, such as Lucene and Elasticsearch. retriv also implements the classic relevance model TF-IDF for educational purposes.

        Args:
            index_name (str, optional): [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index. Defaults to "new-index".

            model (str, optional): defines the retrieval model to use for searching (`bm25` or `tf-idf`). Defaults to "bm25".

            min_df (int, optional): terms that appear in less than `min_df` documents will be ignored. If integer, the parameter indicates the absolute count. If float, it represents a proportion of documents. Defaults to 1.

            tokenizer (Union[str, callable], optional): [tokenizer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable tokenizer or disable tokenization by setting the parameter to `None`. Defaults to "whitespace".

            stemmer (Union[str, callable], optional): [stemmer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable stemmer or disable stemming setting the parameter to `None`. Defaults to "english".

            stopwords (Union[str, List[str], Set[str]], optional): [stopwords](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to remove during preprocessing. You can pass a custom stop-word list or disable stop-words removal by setting the parameter to `None`. Defaults to "english".

            do_lowercasing (bool, optional): whether or not to lowercase texts. Defaults to True.

            do_ampersand_normalization (bool, optional): whether to convert `&` in `and` during pre-processing. Defaults to True.

            do_special_chars_normalization (bool, optional): whether to remove special characters for letters, e.g., `übermensch` → `ubermensch`. Defaults to True.

            do_acronyms_normalization (bool, optional): whether to remove full stop symbols from acronyms without splitting them in multiple words, e.g., `P.C.I.` → `PCI`. Defaults to True.

            do_punctuation_removal (bool, optional): whether to remove punctuation. Defaults to True.

            hyperparams (dict, optional): Retrieval model hyperparams. If `None`, it is automatically set to `{b: 0.75, k1: 1.2}`. Defaults to None.

        Returns:
            SparseRetriever: Sparse Retriever.
        """

        assert model.lower() in {"bm25", "tf-idf"}
        assert min_df > 0, "`min_df` must be greater than zero."
        self.init_args = {
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
        self.inverted_index = None
        self.vocabulary = None
        self.doc_count = None
        self.doc_lens = None
        self.avg_doc_len = None
        self.relative_doc_lens = None
        self.doc_index = None

        self.preprocessing_args = {
            "tokenizer": self.tokenizer,
            "stemmer": self.stemmer,
            "stopwords": self.stopwords,
            "do_lowercasing": self.do_lowercasing,
            "do_ampersand_normalization": self.do_ampersand_normalization,
            "do_special_chars_normalization": self.do_special_chars_normalization,
            "do_acronyms_normalization": self.do_acronyms_normalization,
            "do_punctuation_removal": self.do_punctuation_removal,
        }

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

        np.savez_compressed(sr_state_path(self.index_name), state=state)

    @staticmethod
    def load(index_name: str = "new-index"):
        """Load a retriever and its index.

        Args:
            index_name (str, optional): Name of the index. Defaults to "new-index".

        Returns:
            SparseRetriever: Sparse Retriever.
        """

        state = np.load(sr_state_path(index_name), allow_pickle=True)["state"][()]

        se = SparseRetriever(**state["init_args"])
        se.initialize_doc_index()
        se.id_mapping = state["id_mapping"]
        se.doc_count = state["doc_count"]
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

    def index_aux(self, show_progress: bool = True):
        """Internal usage."""
        collection = read_jsonl(
            docs_path(self.index_name),
            generator=True,
            callback=lambda x: x["text"],
        )

        # Preprocessing --------------------------------------------------------
        collection = multi_preprocessing(
            collection=collection,
            **self.preprocessing_args,
            n_threads=os.cpu_count(),
        )  # This is a generator

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
        """Index a given collection od documents.

        Args:
            collection (Iterable): collection of documents to index.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            SparseRetriever: Sparse Retriever.
        """
        self.save_collection(collection, callback, show_progress)
        self.initialize_doc_index()
        self.initialize_id_mapping()
        self.doc_count = len(self.id_mapping)
        self.index_aux(show_progress)
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
            _type_: _description_
        """
        collection = self.collection_generator(path=path, callback=callback)
        return self.index(collection=collection, show_progress=show_progress)

    # SEARCH ===================================================================
    def query_preprocessing(self, query: str) -> List[str]:
        """Internal usage."""
        return preprocessing(query, **self.preprocessing_args)

    def get_term_doc_freqs(self, query_terms: List[str]) -> nb.types.List:
        """Internal usage."""
        return TypedList([self.inverted_index[t]["tfs"] for t in query_terms])

    def get_doc_ids(self, query_terms: List[str]) -> nb.types.List:
        """Internal usage."""
        return TypedList([self.inverted_index[t]["doc_ids"] for t in query_terms])

    def search(self, query: str, return_docs: bool = True, cutoff: int = 100) -> List:
        """Standard search functionality.

        Args:
            query (str): what to search for.

            return_docs (bool, optional): wether to return the texts of the documents. Defaults to True.

            cutoff (int, optional): number of results to return. Defaults to 100.

        Raises:
            NotImplementedError: _description_

        Returns:
            List: results.
        """
        query_terms = self.query_preprocessing(query)
        if not query_terms:
            return {}
        query_terms = [t for t in query_terms if t in self.vocabulary]
        if not query_terms:
            return {}

        doc_ids = self.get_doc_ids(query_terms)
        term_doc_freqs = self.get_term_doc_freqs(query_terms)

        if self.model == "bm25":
            unique_doc_ids, scores = bm25(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                relative_doc_lens=self.relative_doc_lens,
                doc_count=self.doc_count,
                cutoff=cutoff,
                **self.hyperparams,
            )
        elif self.model == "tf-idf":
            unique_doc_ids, scores = tf_idf(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                doc_lens=self.doc_lens,
                cutoff=cutoff,
            )
        else:
            raise NotImplementedError()

        unique_doc_ids = self.map_internal_ids_to_original_ids(unique_doc_ids)

        if not return_docs:
            return dict(zip(unique_doc_ids, scores))

        return self.prepare_results(unique_doc_ids, scores)

    def msearch(self, queries: List[Dict[str, str]], cutoff: int = 100) -> Dict:
        """Compute results for multiple queries at once.

        Args:
            queries (List[Dict[str, str]]): what to search for.

            cutoff (int, optional): number of results to return. Defaults to 100.

        Returns:
            Dict: results.
        """
        term_doc_freqs = TypedList()
        doc_ids = TypedList()
        q_ids = []
        no_results_q_ids = []

        for q in queries:
            q_id, query = q["id"], q["text"]
            query_terms = self.query_preprocessing(query)
            query_terms = [t for t in query_terms if t in self.vocabulary]
            if not query_terms:
                no_results_q_ids.append(q_id)
                continue

            if all(t not in self.inverted_index for t in query_terms):
                no_results_q_ids.append(q_id)
                continue

            q_ids.append(q_id)
            term_doc_freqs.append(self.get_term_doc_freqs(query_terms))
            doc_ids.append(self.get_doc_ids(query_terms))

        if not q_ids:
            return {q_id: {} for q_id in [q["id"] for q in queries]}

        if self.model == "bm25":
            unique_doc_ids, scores = bm25_multi(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                relative_doc_lens=self.relative_doc_lens,
                doc_count=self.doc_count,
                cutoff=cutoff,
                **self.hyperparams,
            )
        elif self.model == "tf-idf":
            unique_doc_ids, scores = tf_idf_multi(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                doc_lens=self.doc_lens,
                cutoff=cutoff,
            )
        else:
            raise NotImplementedError()

        unique_doc_ids = [
            self.map_internal_ids_to_original_ids(_unique_doc_ids)
            for _unique_doc_ids in unique_doc_ids
        ]

        results = {
            q: dict(zip(unique_doc_ids[i], scores[i])) for i, q in enumerate(q_ids)
        }

        for q_id in no_results_q_ids:
            results[q_id] = {}

        # Order as queries
        return {q_id: results[q_id] for q_id in [q["id"] for q in queries]}

    def bsearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 1_000,
        show_progress: bool = True,
        qrels: Dict[str, Dict[str, float]] = None,
        path: str = None,
    ):
        """Batch-Search is similar to Multi-Search but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format. bsearch is handy for computing results for hundreds of thousands or even millions of queries without hogging your RAM.

        Args:
            queries (List[Dict[str, str]]): what to search for.

            cutoff (int, optional): number of results to return. Defaults to 100.

            batch_size (int, optional): number of query to perform simultaneously. Defaults to 1_000.

            show_progress (bool, optional): whether to show a progress bar for the search process. Defaults to True.

            qrels (Dict[str, Dict[str, float]], optional): query relevance judgements for the queries. Defaults to None.

            path (str, optional): where to save the results. Defaults to None.

        Returns:
            Dict: results.
        """
        batches = [
            queries[i : i + batch_size] for i in range(0, len(queries), batch_size)
        ]

        results = {}

        pbar = tqdm(
            total=len(queries),
            disable=not show_progress,
            desc="Batch search",
            dynamic_ncols=True,
            mininterval=0.5,
        )

        if path is None:
            for batch in batches:
                new_results = self.msearch(queries=batch, cutoff=cutoff)
                results = {**results, **new_results}
                pbar.update(min(batch_size, len(batch)))
        else:
            path = create_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                for batch in batches:
                    new_results = self.msearch(queries=batch, cutoff=cutoff)

                    for i, (k, v) in enumerate(new_results.items()):
                        x = {
                            "id": k,
                            "text": batch[i]["text"],
                            f"{self.model}_doc_ids": list(v.keys()),
                            f"{self.model}_scores": [
                                float(s) for s in list(v.values())
                            ],
                        }
                        if qrels is not None:
                            x["rel_doc_ids"] = list(qrels[k].keys())
                            x["rel_scores"] = list(qrels[k].values())
                        f.write(orjson.dumps(x) + "\n".encode())

                    pbar.update(min(batch_size, len(batch)))

        return results

    def autotune(
        self,
        queries: List[Dict[str, str]],
        qrels: Dict[str, Dict[str, float]],
        metric: str = "ndcg",
        n_trials: int = 100,
        cutoff: int = 100,
    ):
        """Use the AutoTune function to tune [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) parameters w.r.t. your document collection and queries.
        All metrics supported by [ranx](https://github.com/AmenRa/ranx) are supported by the `autotune` function.
        At the of the process, the best parameter configuration is automatically applied to the `SparseRetriever` instance and saved to disk.
        You can inspect the current configuration by printing `sr.hyperparams`.

        Args:
            queries (List[Dict[str, str]]): queries to use for the optimization process.

            qrels (Dict[str, Dict[str, float]]): query relevance judgements for the queries.

            metric (str, optional): metric to optimize for. Defaults to "ndcg".

            n_trials (int, optional): number of configuration to evaluate. Defaults to 100.

            cutoff (int, optional): number of results to consider for the optimization process. Defaults to 100.
        """
        hyperparams = tune_bm25(
            queries=queries,
            qrels=qrels,
            se=self,
            metric=metric,
            n_trials=n_trials,
            cutoff=cutoff,
        )
        self.hyperparams = hyperparams
        self.save()
