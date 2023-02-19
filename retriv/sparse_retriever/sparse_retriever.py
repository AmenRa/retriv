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

        self.hyperparams = (
            dict(b=0.75, k1=1.2) if hyperparams is None else hyperparams
        )

    def save(self):
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
        state = np.load(sr_state_path(index_name), allow_pickle=True)["state"][
            ()
        ]

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
        self.save_collection(collection, callback, show_progress)
        self.initialize_doc_index()
        self.initialize_id_mapping()
        self.doc_count = len(self.id_mapping)
        self.index_aux(show_progress)
        self.save()
        return self

    def index_file(
        self, path: str, callback: callable = None, show_progress: bool = True
    ) -> None:
        collection = self.collection_generator(path=path, callback=callback)
        return self.index(collection=collection, show_progress=show_progress)

    # SEARCH ===================================================================
    def query_preprocessing(self, query: str) -> List[str]:
        return preprocessing(query, **self.preprocessing_args)

    def get_term_doc_freqs(self, query_terms: List[str]) -> nb.types.List:
        return TypedList([self.inverted_index[t]["tfs"] for t in query_terms])

    def get_doc_ids(self, query_terms: List[str]) -> nb.types.List:
        return TypedList(
            [self.inverted_index[t]["doc_ids"] for t in query_terms]
        )

    def search(self, query: str, return_docs: bool = True, cutoff: int = 100):
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

    def msearch(self, queries: List[Dict[str, str]], cutoff: int = 100):
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
            q: dict(zip(unique_doc_ids[i], scores[i]))
            for i, q in enumerate(q_ids)
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
        batches = [
            queries[i : i + batch_size]
            for i in range(0, len(queries), batch_size)
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
