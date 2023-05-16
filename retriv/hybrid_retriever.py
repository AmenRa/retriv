from typing import Dict, Iterable, List, Set, Union

import numpy as np
import orjson
from oneliner_utils import create_path
from tqdm import tqdm

from .base_retriever import BaseRetriever
from .dense_retriever.dense_retriever import DenseRetriever
from .merger.merger import Merger
from .paths import hr_state_path
from .sparse_retriever.sparse_retriever import SparseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        # Global params
        index_name: str = "new-index",
        # Sparse retriever params
        sr_model: str = "bm25",
        min_df: int = 1,
        tokenizer: Union[str, callable] = "whitespace",
        stemmer: Union[str, callable] = "english",
        stopwords: Union[str, List[str], Set[str]] = "english",
        do_lowercasing: bool = True,
        do_ampersand_normalization: bool = True,
        do_special_chars_normalization: bool = True,
        do_acronyms_normalization: bool = True,
        do_punctuation_removal: bool = True,
        # Dense retriever params
        dr_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        max_length: int = 128,
        use_ann: bool = True,
        # For already instantiated modules
        sparse_retriever: SparseRetriever = None,
        dense_retriever: DenseRetriever = None,
        merger: Merger = None,
    ):
        self.index_name = index_name

        self.sparse_retriever = (
            sparse_retriever
            if sparse_retriever is not None
            else SparseRetriever(
                index_name=index_name,
                model=sr_model,
                min_df=min_df,
                tokenizer=tokenizer,
                stemmer=stemmer,
                stopwords=stopwords,
                do_lowercasing=do_lowercasing,
                do_ampersand_normalization=do_ampersand_normalization,
                do_special_chars_normalization=do_special_chars_normalization,
                do_acronyms_normalization=do_acronyms_normalization,
                do_punctuation_removal=do_punctuation_removal,
            )
        )

        self.dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else DenseRetriever(
                index_name=index_name,
                model=dr_model,
                normalize=normalize,
                max_length=max_length,
                use_ann=use_ann,
            )
        )

        self.merger = merger if merger is not None else Merger(index_name=index_name)

    def index(
        self,
        collection: Iterable,
        embeddings_path: str = None,
        use_gpu: bool = False,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ):
        self.save_collection(collection, callback, show_progress)

        self.initialize_doc_index()
        self.initialize_id_mapping()
        self.doc_count = len(self.id_mapping)

        # Sparse ---------------------------------------------------------------
        self.sparse_retriever.doc_index = self.doc_index
        self.sparse_retriever.id_mapping = self.id_mapping
        self.sparse_retriever.doc_count = self.doc_count
        self.sparse_retriever.index_aux(show_progress)

        # Dense ----------------------------------------------------------------
        self.dense_retriever.doc_index = self.doc_index
        self.dense_retriever.id_mapping = self.id_mapping
        self.dense_retriever.doc_count = self.doc_count
        self.dense_retriever.index_aux(
            embeddings_path, use_gpu, batch_size, callback, show_progress
        )

        self.save()

        return self

    def index_file(
        self,
        path: str,
        embeddings_path: str = None,
        use_gpu: bool = False,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ) -> None:
        collection = self.collection_generator(path, callback)
        return self.index(
            collection,
            embeddings_path,
            use_gpu,
            batch_size,
            None,
            show_progress,
        )

    def save(self):
        state = dict(
            id_mapping=self.id_mapping,
            doc_count=self.doc_count,
        )
        np.savez_compressed(hr_state_path(self.index_name), state=state)

        self.sparse_retriever.save()
        self.dense_retriever.save()
        self.merger.save()

    @staticmethod
    def load(index_name: str = "new-index"):
        state = np.load(hr_state_path(index_name), allow_pickle=True)["state"][()]

        hr = HybridRetriever(index_name)
        hr.initialize_doc_index()
        hr.id_mapping = state["id_mapping"]
        hr.doc_count = state["doc_count"]

        hr.sparse_retriever = SparseRetriever.load(index_name)
        hr.dense_retriever = DenseRetriever.load(index_name)
        hr.merger = Merger.load(index_name)
        return hr

    def search(
        self,
        query: str,
        return_docs: bool = True,
        cutoff: int = 100,
    ):
        sparse_results = self.sparse_retriever.search(query, False, 1_000)
        dense_results = self.dense_retriever.search(query, False, 1_000)
        hybrid_results = self.merger.fuse([sparse_results, dense_results])
        return (
            self.prepare_results(
                list(hybrid_results.keys())[:cutoff],
                list(hybrid_results.values())[:cutoff],
            )
            if return_docs
            else hybrid_results
        )

    def msearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
    ):
        sparse_results = self.sparse_retriever.msearch(queries, 1_000)
        dense_results = self.dense_retriever.msearch(queries, 1_000, batch_size)
        return self.merger.mfuse([sparse_results, dense_results], cutoff)

    def bsearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
        show_progress: bool = True,
        qrels: Dict[str, Dict[str, float]] = None,
        path: str = None,
    ):
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
                new_results = self.msearch(
                    queries=batch, cutoff=cutoff, batch_size=len(batch)
                )
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
                            "hybrid_doc_ids": list(v.keys()),
                            "hybrid_scores": [float(s) for s in list(v.values())],
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
        batch_size: int = 32,
    ):
        # Tune sparse ----------------------------------------------------------
        self.sparse_retriever.autotune(
            queries=queries,
            qrels=qrels,
            metric=metric,
            n_trials=n_trials,
            cutoff=cutoff,
        )

        # Tune merger ----------------------------------------------------------
        sparse_results = self.sparse_retriever.msearch(queries, 1_000)
        dense_results = self.dense_retriever.msearch(queries, 1_000, batch_size)
        self.merger.autotune(qrels, [sparse_results, dense_results], metric)
