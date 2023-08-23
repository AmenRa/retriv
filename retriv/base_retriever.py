import os
import shutil
from typing import Iterable, List

import numpy as np
import orjson
from indxr import Indxr
from oneliner_utils import read_csv, read_jsonl

from .paths import docs_path, index_path


class BaseRetriever:
    def __init__(self, index_name: str = "new-index"):
        self.index_name = index_name
        self.id_mapping = None
        self.doc_count = None
        self.doc_index = None

    @staticmethod
    def delete(index_name="new-index"):
        try:
            shutil.rmtree(index_path(index_name))
            print(f"{index_name} successfully removed.")
        except FileNotFoundError:
            print(f"{index_name} not found.")

    def collection_generator(self, path: str, callback: callable = None):
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

    def save_collection(self, collection: Iterable, callback: callable = None):
        with open(docs_path(self.index_name), "wb") as f:
            for doc in collection:
                x = callback(doc) if callback is not None else doc
                f.write(orjson.dumps(x) + "\n".encode())

    def initialize_doc_index(self):
        self.doc_index = Indxr(docs_path(self.index_name))

    def initialize_id_mapping(self):
        ids = read_jsonl(
            docs_path(self.index_name),
            generator=True,
            callback=lambda x: x["id"],
        )
        self.id_mapping = dict(enumerate(ids))

    def get_doc(self, doc_id: str) -> dict:
        return self.doc_index.get(doc_id)

    def get_docs(self, doc_ids: List[str]) -> List[dict]:
        return self.doc_index.mget(doc_ids)

    def prepare_results(self, doc_ids: List[str], scores: np.ndarray) -> List[dict]:
        docs = self.get_docs(doc_ids)
        results = []
        for doc, score in zip(docs, scores):
            doc["score"] = score
            results.append(doc)

        return results

    def map_internal_ids_to_original_ids(self, doc_ids: Iterable) -> List[str]:
        return [self.id_mapping[doc_id] for doc_id in doc_ids]

    def save(self):
        raise NotImplementedError()

    @staticmethod
    def load():
        raise NotImplementedError()

    def index(self):
        raise NotImplementedError()

    def index_file(self):
        raise NotImplementedError()

    def search(self):
        raise NotImplementedError()

    def msearch(self):
        raise NotImplementedError()

    def autotune(self):
        raise NotImplementedError()
