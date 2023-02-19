import os
import shutil
from typing import Dict, Iterable, List

import numpy as np
import orjson
from numba import njit, prange
from numba.typed import List as TypedList
from oneliner_utils import create_path
from tqdm import tqdm

from ..base_retriever import BaseRetriever
from ..paths import docs_path, dr_state_path, embeddings_folder_path
from .ann_searcher import ANN_Searcher
from .encoder import Encoder


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        index_name: str = "new-index",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        max_length: int = 128,
        use_ann: bool = True,
    ):
        self.index_name = index_name
        self.model = model
        self.normalize = normalize
        self.max_length = max_length
        self.use_ann = use_ann

        self.encoder = Encoder(
            index_name=index_name,
            model=model,
            normalize=normalize,
            max_length=max_length,
        )

        self.ann_searcher = ANN_Searcher(index_name=index_name)

        self.id_mapping = None
        self.doc_count = None
        self.doc_index = None

        self.embeddings = None

    def save(self):
        state = dict(
            init_args=dict(
                index_name=self.index_name,
                model=self.model,
                normalize=self.normalize,
                max_length=self.max_length,
                use_ann=self.use_ann,
            ),
            id_mapping=self.id_mapping,
            doc_count=self.doc_count,
            embeddings=True if self.embeddings is not None else None,
        )
        np.savez_compressed(dr_state_path(self.index_name), state=state)

    @staticmethod
    def load(index_name: str = "new-index"):
        state = np.load(dr_state_path(index_name), allow_pickle=True)["state"][
            ()
        ]
        dr = DenseRetriever(**state["init_args"])
        dr.initialize_doc_index()
        dr.id_mapping = state["id_mapping"]
        dr.doc_count = state["doc_count"]
        if state["embeddings"]:
            dr.load_embeddings()
        if dr.use_ann:
            dr.ann_searcher = ANN_Searcher.load(index_name)
        return dr

    def load_embeddings(self):
        path = embeddings_folder_path(self.index_name)
        npy_file_paths = sorted(os.listdir(path))
        self.embeddings = np.concatenate(
            [np.load(path / npy_file_path) for npy_file_path in npy_file_paths]
        )

    def import_embeddings(self, path: str):
        shutil.copyfile(
            path, embeddings_folder_path(self.index_name) / "chunk_0.npy"
        )

    def index_aux(
        self,
        embeddings_path: str = None,
        use_gpu: bool = False,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ):
        if embeddings_path is not None:
            self.import_embeddings(embeddings_path)
        else:
            self.encoder.change_device("cuda" if use_gpu else "cpu")
            self.encoder.encode_collection(
                path=docs_path(self.index_name),
                batch_size=batch_size,
                callback=callback,
                show_progress=show_progress,
            )
            self.encoder.change_device("cpu")

        if self.use_ann:
            if show_progress:
                print("Building ANN Searcher")
            self.ann_searcher.build()
        else:
            if show_progress:
                print("Loading embeddings...")
            self.load_embeddings()

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
        self.index_aux(
            embeddings_path=embeddings_path,
            use_gpu=use_gpu,
            batch_size=batch_size,
            callback=callback,
            show_progress=show_progress,
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

    def search(
        self,
        query: str,
        return_docs: bool = True,
        cutoff: int = 100,
    ):
        encoded_query = self.encoder(query)

        if self.use_ann:
            doc_ids, scores = self.ann_searcher.search(encoded_query, cutoff)
        else:
            if self.embeddings is None:
                self.load_embeddings()
            doc_ids, scores = compute_scores(
                encoded_query, self.embeddings, cutoff
            )

        doc_ids = self.map_internal_ids_to_original_ids(doc_ids)

        return (
            self.prepare_results(doc_ids, scores)
            if return_docs
            else dict(zip(doc_ids, scores))
        )

    def msearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
    ):
        q_ids = [x["id"] for x in queries]
        q_texts = [x["text"] for x in queries]
        encoded_queries = self.encoder(q_texts, batch_size, show_progress=False)

        if self.use_ann:
            doc_ids, scores = self.ann_searcher.msearch(encoded_queries, cutoff)
        else:
            if self.embeddings is None:
                self.load_embeddings()
            doc_ids, scores = compute_scores_multi(
                encoded_queries, self.embeddings, cutoff
            )

        doc_ids = [
            self.map_internal_ids_to_original_ids(_doc_ids)
            for _doc_ids in doc_ids
        ]

        results = {
            q: dict(zip(doc_ids[i], scores[i])) for i, q in enumerate(q_ids)
        }

        return {q_id: results[q_id] for q_id in q_ids}

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
                            "dense_doc_ids": list(v.keys()),
                            "dense_scores": [
                                float(s) for s in list(v.values())
                            ],
                        }
                        if qrels is not None:
                            x["rel_doc_ids"] = list(qrels[k].keys())
                            x["rel_scores"] = list(qrels[k].values())
                        f.write(orjson.dumps(x) + "\n".encode())

                    pbar.update(min(batch_size, len(batch)))

        return results


@njit(cache=True)
def compute_scores(query: np.ndarray, docs: np.ndarray, cutoff: int):
    scores = docs @ query
    indices = np.argsort(-scores)[:cutoff]

    return indices, scores[indices]


@njit(cache=True, parallel=True)
def compute_scores_multi(queries: np.ndarray, docs: np.ndarray, cutoff: int):
    n = len(queries)
    ids = TypedList([np.empty(1, dtype=np.int64) for _ in range(n)])
    scores = TypedList([np.empty(1, dtype=np.float32) for _ in range(n)])

    for i in prange(len(queries)):
        ids[i], scores[i] = compute_scores(queries[i], docs, cutoff)

    return ids, scores
