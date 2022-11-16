import os
import shutil
import string
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Union

import numba as nb
import numpy as np
import orjson
from indxr import Indxr
from numba.typed import List as TypedList
from oneliner_utils import read_csv, read_jsonl
from rich.progress import track

from .autotune import tune_bm25
from .bm25 import bm25, bm25_multi
from .stemmers import get_stemmer
from .stopwords import get_stopwords
from .tokenizers import get_tokenizer


def home_path():
    p = Path(Path.home() / ".collections")
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepro(x: str, tokenizer: Callable, sw_list: list, stemmer: Callable):
    x = x.lower()
    x = x.translate(str.maketrans("", "", string.punctuation))  # Remove punct
    x = tokenizer(x)
    x = [t for t in x if t not in sw_list]

    return [stemmer(t) for t in x]


def prepro_caller(input):
    return prepro(*input)


class SearchEngine:
    def __init__(
        self,
        index_name: str = "new-index",
        min_term_freq: int = 1,
        tokenizer: Union[str, Callable] = "whitespace",
        stemmer: Union[str, Callable] = "english",
        sw_list: Union[str, List] = "english",
    ):
        assert min_term_freq > 0, "`min_term_freq` must be greater than zero."
        self.init_args = {
            "index_name": index_name,
            "tokenizer": tokenizer,
            "stemmer": stemmer,
            "sw_list": sw_list,
        }
        self.min_term_freq = min_term_freq

        self.index_name = index_name
        self.index_path = home_path() / index_name
        self.index_path.mkdir(parents=True, exist_ok=True)

        if type(tokenizer) is str:
            self.tokenizer = get_tokenizer(tokenizer)
        elif callable(tokenizer):
            self.tokenizer = tokenizer
        elif tokenizer is False:
            self.tokenizer = lambda x: x
        else:
            raise (NotImplementedError)

        if type(stemmer) is str:
            self.stemmer = get_stemmer(stemmer)
        elif callable(stemmer):
            self.stemmer = stemmer
        elif stemmer is False:
            self.stemmer = lambda x: x
        else:
            raise (NotImplementedError)

        if type(sw_list) is str:
            self.sw_list = get_stopwords(sw_list)
        elif type(sw_list) is list:
            self.sw_list = sw_list
        elif sw_list is False:
            self.sw_list = []
        else:
            raise (NotImplementedError)

        self.id_mapping = None
        self.inverted_index = None
        self.doc_count = None
        self.doc_lens = None
        self.relative_doc_lens = None
        self.docs_path = None
        self.doc_index = None

    def index(
        self,
        collection: List[Dict],
        show_progress: bool = True,
    ):
        self.docs_path = str(self.index_path / "docs.jsonl")

        with open(self.docs_path, "wb") as f:
            for y in track(
                collection,
                disable=not show_progress,
                description="Saving collection",
            ):
                f.write(orjson.dumps(y) + "\n".encode())

        # write_jsonl(collection, self.docs_path)
        self.doc_index = Indxr(self.docs_path)

        self.id_mapping = {i: x["id"] for i, x in enumerate(collection)}
        self.doc_count = len(self.id_mapping)

        # Preprocessing --------------------------------------------------------
        collection = self.preprocessing(collection, show_progress)

        # Inverted index -------------------------------------------------------
        self.inverted_index = self.build_inverted_index(
            collection, show_progress
        )

        # Pre-compute some stuff -----------------------------------------------
        self.doc_lens = self.get_doc_lens(collection)
        self.relative_doc_lens = self.get_relative_doc_lens(self.doc_lens)

        self.save()

    def index_file(
        self,
        path: str,
        show_progress: bool = True,
        callback: Callable = None,
    ):
        kind = os.path.splitext(path)[1][1:]
        assert kind in {
            "jsonl",
            "csv",
            "tsv",
        }, "Only JSONl, CSV, and TSV are currently supported."

        if kind == "jsonl":
            collection = read_jsonl(path, callback=callback)
        elif kind == "csv":
            collection = read_csv(path, callback=callback)
        elif kind == "tsv":
            collection = read_csv(path, delimiter="\t", callback=callback)

        self.index(
            collection=collection,
            show_progress=show_progress,
        )

    def save(self):
        state = {
            "init_args": self.init_args,
            "docs_path": self.docs_path,
            "id_mapping": self.id_mapping,
            "doc_count": self.doc_count,
            "inverted_index": self.inverted_index,
            "doc_lens": self.doc_lens,
            "relative_doc_lens": self.relative_doc_lens,
        }

        np.save(self.index_path / "index.npy", state)

    @staticmethod
    def load(index_name="new-index"):
        index_path = home_path() / index_name / "index.npy"
        state = np.load(index_path, allow_pickle=True)[()]
        se = SearchEngine(**state["init_args"])
        se.docs_path = state["docs_path"]
        se.doc_index = Indxr(se.docs_path)
        se.id_mapping = state["id_mapping"]
        se.doc_count = state["doc_count"]
        se.inverted_index = state["inverted_index"]
        se.doc_lens = state["doc_lens"]
        se.relative_doc_lens = state["relative_doc_lens"]

        return se

    @staticmethod
    def delete(index_name="new-index"):
        try:
            shutil.rmtree(home_path() / index_name)
            print(f"{index_name} successfully removed.")
        except FileNotFoundError:
            print(f"{index_name} not found.")

    def preprocessing(self, collection, show_progress: bool = True):
        inputs = [
            (
                x["contents"],
                self.tokenizer,
                self.sw_list,
                self.stemmer,
            )
            for x in collection
        ]

        with Pool() as p:
            preprocessed = list(
                track(
                    p.imap(
                        prepro_caller,
                        inputs,
                        chunksize=1_000,
                    ),
                    total=self.doc_count,
                    disable=not show_progress,
                    description="Processing texts",
                )
            )

        for x, y in zip(collection, preprocessed):
            x["contents"] = y

        return collection

    def get_doc_lens(self, collection):
        return [len(x["contents"]) for x in collection]

    def get_relative_doc_lens(self, doc_lens):
        return (doc_lens / np.mean(doc_lens)).astype(np.float32)

    def build_inverted_index(self, collection, show_progress: bool = True):
        inverted_index = defaultdict(lambda: defaultdict(list))

        for i, doc in enumerate(
            track(
                collection,
                disable=not show_progress,
                description="Building inverted index",
            )
        ):
            unique, counts = np.unique(doc["contents"], return_counts=True)

            for term, occurrences in zip(unique, counts):
                inverted_index[term]["doc_ids"].append(i)
                inverted_index[term]["tfs"].append(occurrences)

        for term in track(
            inverted_index.keys(),
            total=len(inverted_index.keys()),
            disable=not show_progress,
            description="Optimizing inverted index",
        ):
            if len(inverted_index[term]["doc_ids"]) < self.min_term_freq:
                del inverted_index[term]

            inverted_index[term]["doc_ids"] = np.array(
                inverted_index[term]["doc_ids"], dtype=np.int32
            )
            inverted_index[term]["tfs"] = np.array(
                inverted_index[term]["tfs"], dtype=np.int16
            )
            inverted_index[term] = dict(inverted_index[term])

        return dict(inverted_index)

    def get_term_doc_freqs(self, query_terms: List[str]) -> nb.types.List:
        return TypedList(
            [
                self.inverted_index[t]["tfs"]
                for t in query_terms
                if t in self.inverted_index
            ]
        )

    def get_doc_ids(self, query_terms: List[str]) -> nb.types.List:
        return TypedList(
            [
                self.inverted_index[t]["doc_ids"]
                for t in query_terms
                if t in self.inverted_index
            ]
        )

    def search(
        self,
        query: str,
        return_docs: bool = True,
        b: float = 0.75,
        k1: float = 1.2,
        n_res: int = 100,
    ):
        query_terms = prepro(query, self.tokenizer, self.sw_list, self.stemmer)
        term_doc_freqs = self.get_term_doc_freqs(query_terms)
        doc_ids = self.get_doc_ids(query_terms)

        unique_doc_ids, scores = bm25(
            b=b,
            k1=k1,
            term_doc_freqs=term_doc_freqs,
            doc_ids=doc_ids,
            relative_doc_lens=self.relative_doc_lens,
            n_res=n_res,
        )

        unique_doc_ids = [self.id_mapping[doc_id] for doc_id in unique_doc_ids]

        if return_docs:
            docs = self.doc_index.mget(unique_doc_ids)
            results = []
            for doc, score in zip(docs, scores):
                doc["score"] = score
                results.append(doc)
            return results
        else:
            return dict(zip(unique_doc_ids, scores))

    def msearch(
        self,
        queries: Dict[str, str],
        b: float = 0.75,
        k1: float = 1.2,
        n_res: int = 100,
    ):

        term_doc_freqs = TypedList()
        doc_ids = TypedList()
        q_ids = []
        no_results_q_ids = []

        for q_id, query in queries.items():
            query_terms = prepro(
                query, self.tokenizer, self.sw_list, self.stemmer
            )

            if all(t not in self.inverted_index for t in query_terms):
                no_results_q_ids.append(q_id)
                continue

            q_ids.append(q_id)
            term_doc_freqs.append(self.get_term_doc_freqs(query_terms))
            doc_ids.append(self.get_doc_ids(query_terms))

        unique_doc_ids, scores = bm25_multi(
            b=b,
            k1=k1,
            term_doc_freqs=term_doc_freqs,
            doc_ids=doc_ids,
            relative_doc_lens=self.relative_doc_lens,
            n_res=n_res,
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
        return {q_id: results[q_id] for q_id in queries}

    def autotune(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, float]],
        metric: str = "ndcg@100",
        n_trials: int = 100,
        n_res: int = 100,
    ):
        return tune_bm25(
            queries=queries,
            qrels=qrels,
            se=self,
            metric=metric,
            n_trials=n_trials,
            n_res=n_res,
        )
