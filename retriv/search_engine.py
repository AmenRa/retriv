import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Union

import numba as nb
import numpy as np
import orjson
from indxr import Indxr
from numba.typed import List as TypedList
from oneliner_utils import create_path, read_csv, read_jsonl
from tqdm import tqdm

from .autotune import tune_bm25
from .build_inverted_index import build_inverted_index
from .preprocessing import (
    get_spell_corrector,
    get_stemmer,
    get_stopwords,
    get_tokenizer,
    multi_preprocessing,
    preprocessing,
)
from .retrieval_functions.bm25 import bm25, bm25_multi


def home_path():
    p = Path(Path.home() / ".retriv" / "collections")
    p.mkdir(parents=True, exist_ok=True)
    return p


class SearchEngine:
    def __init__(
        self,
        index_name: str = "new-index",
        min_df: int = 1,
        tokenizer: Union[str, callable] = "whitespace",
        stemmer: Union[str, callable] = "english",
        stopwords: Union[str, List[str], Set[str]] = "english",
        spell_corrector: str = None,
        do_lowercasing: bool = True,
        do_ampersand_normalization: bool = True,
        do_special_chars_normalization: bool = True,
        do_acronyms_normalization: bool = True,
        do_punctuation_removal: bool = True,
        hyperparams: dict = None,
    ):
        assert min_df > 0, "`min_df` must be greater than zero."
        self.init_args = {
            "index_name": index_name,
            "do_lowercasing": do_lowercasing,
            "do_ampersand_normalization": do_ampersand_normalization,
            "do_special_chars_normalization": do_special_chars_normalization,
            "do_acronyms_normalization": do_acronyms_normalization,
            "do_punctuation_removal": do_punctuation_removal,
            "tokenizer": tokenizer,
            "stemmer": stemmer,
            "stopwords": stopwords,
            "spell_corrector": spell_corrector,
        }
        self.min_df = min_df

        self.index_name = index_name
        self.index_path = home_path() / index_name
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.do_lowercasing = do_lowercasing
        self.do_ampersand_normalization = do_ampersand_normalization
        self.do_special_chars_normalization = do_special_chars_normalization
        self.do_acronyms_normalization = do_acronyms_normalization
        self.do_punctuation_removal = do_punctuation_removal

        self.tokenizer = get_tokenizer(tokenizer)
        self.stemmer = get_stemmer(stemmer)
        self.stopwords = [self.stemmer(sw) for sw in get_stopwords(stopwords)]
        self.spell_corrector = get_spell_corrector(spell_corrector)

        self.id_mapping = None
        self.inverted_index = None
        self.vocabulary = None
        self.doc_count = None
        self.doc_lens = None
        self.relative_doc_lens = None
        self.docs_path = None
        self.doc_index = None

        self.preprocessing_args = {
            "tokenizer": self.tokenizer,
            "stemmer": self.stemmer,
            "stopwords": self.stopwords,
            "spell_corrector": self.spell_corrector,
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
        print(f"Saving {self.index_name} index on disk...")

        state = {
            "init_args": self.init_args,
            "docs_path": self.docs_path,
            "id_mapping": self.id_mapping,
            "doc_count": self.doc_count,
            "inverted_index": self.inverted_index,
            "vocabulary": self.vocabulary,
            "doc_lens": self.doc_lens,
            "relative_doc_lens": self.relative_doc_lens,
            "hyperparams": self.hyperparams,
        }

        np.save(self.index_path / "index.npy", state)

    @staticmethod
    def load(index_name="new-index"):
        print(f"Loading {index_name} from disk...")

        index_path = home_path() / index_name / "index.npy"
        state = np.load(index_path, allow_pickle=True)[()]
        se = SearchEngine(**state["init_args"])
        se.docs_path = state["docs_path"]
        se.doc_index = Indxr(se.docs_path)
        se.id_mapping = state["id_mapping"]
        se.doc_count = state["doc_count"]
        se.inverted_index = state["inverted_index"]
        se.vocabulary = set(se.inverted_index)
        se.doc_lens = state["doc_lens"]
        se.relative_doc_lens = state["relative_doc_lens"]
        se.hyperparams = state["hyperparams"]

        return se

    @staticmethod
    def delete(index_name="new-index"):
        try:
            shutil.rmtree(home_path() / index_name)
            print(f"{index_name} successfully removed.")
        except FileNotFoundError:
            print(f"{index_name} not found.")

    def collection_generator(
        self,
        path: str,
        callback: callable = None,
    ):
        kind = os.path.splitext(path)[1][1:]
        assert kind in {
            "jsonl",
            "csv",
            "tsv",
        }, "Only JSONl, CSV, and TSV are currently supported."

        if kind == "jsonl":
            collection = read_jsonl(path, generator=True, callback=callback)
        elif kind == "csv":
            collection = read_csv(path, generator=True, callback=callback)
        elif kind == "tsv":
            collection = read_csv(
                path, delimiter="\t", generator=True, callback=callback
            )

        return collection

    def save_collection(
        self,
        collection: Iterable,
        callback: callable = None,
        show_progress: bool = True,
    ):
        if show_progress:
            print("Saving collection...")

        self.docs_path = str(self.index_path / "docs.jsonl")

        with open(self.docs_path, "wb") as f:
            for doc in collection:
                x = callback(doc) if callback is not None else doc
                f.write(orjson.dumps(x) + "\n".encode())

    def initialize_doc_index(self):
        self.doc_index = Indxr(self.docs_path)

    def initialize_id_mapping(self):
        ids = read_jsonl(
            self.docs_path, generator=True, callback=lambda x: x["id"]
        )
        self.id_mapping = dict(enumerate(ids))

    def initialize_id_mapping(self):
        ids = read_jsonl(
            self.docs_path, generator=True, callback=lambda x: x["id"]
        )
        self.id_mapping = dict(enumerate(ids))

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

        collection = read_jsonl(
            self.docs_path, generator=True, callback=lambda x: x["text"]
        )

        # Preprocessing --------------------------------------------------------
        collection = multi_preprocessing(
            collection=collection,
            **self.preprocessing_args,
            n_threads=os.cpu_count(),
        )  # This is a generator

        # Inverted index -------------------------------------------------------
        self.inverted_index, self.relative_doc_lens = build_inverted_index(
            collection=collection,
            n_docs=self.doc_count,
            min_df=self.min_df,
            show_progress=show_progress,
        )
        self.vocabulary = set(self.inverted_index)

        self.save()

    def index_file(
        self, path: str, callback: callable = None, show_progress: bool = True
    ) -> None:
        collection = self.collection_generator(path=path, callback=callback)
        self.index(collection=collection, show_progress=show_progress)

    # SEARCH ===================================================================
    def query_preprocessing(self, query: str) -> List[str]:
        return preprocessing(query, **self.preprocessing_args)

    def get_term_doc_freqs(self, query_terms: List[str]) -> nb.types.List:
        return TypedList([self.inverted_index[t]["tfs"] for t in query_terms])

    def get_doc_ids(self, query_terms: List[str]) -> nb.types.List:
        return TypedList(
            [self.inverted_index[t]["doc_ids"] for t in query_terms]
        )

    def get_doc(self, doc_id: str) -> dict:
        return self.doc_index.get(doc_id)

    def get_docs(self, doc_ids: List[str]) -> List[dict]:
        return self.doc_index.mget(doc_ids)

    def prepare_results(
        self, doc_ids: List[str], scores: np.ndarray
    ) -> List[dict]:
        docs = self.get_docs(doc_ids)
        results = []
        for doc, score in zip(docs, scores):
            doc["score"] = score
            results.append(doc)

        return results

    def map_internal_ids_to_original_ids(self, doc_ids: Iterable) -> List[str]:
        return [self.id_mapping[doc_id] for doc_id in doc_ids]

    def search(
        self,
        query: str,
        return_docs: bool = True,
        cutoff: int = 100,
    ):
        query_terms = self.query_preprocessing(query)
        if not query_terms:
            return {}
        query_terms = [t for t in query_terms if t in self.vocabulary]
        if not query_terms:
            return {}

        doc_ids = self.get_doc_ids(query_terms)
        term_doc_freqs = self.get_term_doc_freqs(query_terms)

        unique_doc_ids, scores = bm25(
            term_doc_freqs=term_doc_freqs,
            doc_ids=doc_ids,
            relative_doc_lens=self.relative_doc_lens,
            cutoff=cutoff,
            **self.hyperparams,
        )

        unique_doc_ids = self.map_internal_ids_to_original_ids(unique_doc_ids)

        if not return_docs:
            return dict(zip(unique_doc_ids, scores))

        return self.prepare_results(unique_doc_ids, scores)

    def msearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
    ):
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

        unique_doc_ids, scores = bm25_multi(
            term_doc_freqs=term_doc_freqs,
            doc_ids=doc_ids,
            relative_doc_lens=self.relative_doc_lens,
            cutoff=cutoff,
            **self.hyperparams,
        )

        unique_doc_ids = [
            [self.id_mapping[doc_id] for doc_id in _unique_doc_ids]
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
        chunksize: int = 1_000,
        show_progress=True,
        qrels: Dict[str, Dict[str, float]] = None,
        path: str = None,
    ):
        chunks = [
            queries[i : i + chunksize]
            for i in range(0, len(queries), chunksize)
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
            for chunk in chunks:
                new_results = self.msearch(queries=chunk, cutoff=cutoff)
                results = {**results, **new_results}
                pbar.update(min(chunksize, len(chunk)))
        else:
            path = create_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                for chunk in chunks:
                    new_results = self.msearch(queries=chunk, cutoff=cutoff)

                    for i, (k, v) in enumerate(new_results.items()):
                        x = {
                            "id": k,
                            "text": chunk[i]["text"],
                            "bm25_doc_ids": list(v.keys()),
                            "bm25_scores": [float(s) for s in list(v.values())],
                        }
                        if qrels is not None:
                            x["rel_doc_ids"] = list(qrels[k].keys())
                            x["rel_scores"] = list(qrels[k].values())
                        f.write(orjson.dumps(x) + "\n".encode())

                    pbar.update(min(chunksize, len(chunk)))

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
