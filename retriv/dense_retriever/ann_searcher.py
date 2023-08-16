import math

import faiss
import numpy as np
import psutil
from autofaiss import build_index
from oneliner_utils import read_json

from ..paths import embeddings_folder_path, faiss_index_infos_path, faiss_index_path


def get_ram():
    size_bytes = psutil.virtual_memory().total
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s}GB"


class ANN_Searcher:
    def __init__(self, index_name: str = "new-index"):
        self.index_name = index_name
        self.faiss_index = None
        self.faiss_index_infos = None

    def build(self, use_gpu=False):
        index, index_infos = build_index(
            embeddings=str(embeddings_folder_path(self.index_name)),
            index_path=str(faiss_index_path(self.index_name)),
            index_infos_path=str(faiss_index_infos_path(self.index_name)),
            save_on_disk=True,
            metric_type="ip",
            # max_index_memory_usage="32GB",
            current_memory_available=get_ram(),
            max_index_query_time_ms=10,
            min_nearest_neighbors_to_retrieve=20,
            index_key=None,
            index_param=None,
            use_gpu=use_gpu,
            nb_cores=None,
            make_direct_map=False,
            should_be_memory_mappable=False,
            distributed=None,
            verbose=40,
        )

        self.faiss_index = index
        self.faiss_index_infos = index_infos

    @staticmethod
    def load(index_name: str = "new-index"):
        ann_searcher = ANN_Searcher(index_name)
        ann_searcher.faiss_index = faiss.read_index(str(faiss_index_path(index_name)))
        ann_searcher.faiss_index_infos = read_json(faiss_index_infos_path(index_name))
        return ann_searcher

    def search(self, query: np.ndarray, cutoff: int = 100):
        query = query.reshape(1, len(query))
        ids, scores = self.msearch(query, cutoff)
        return ids[0], scores[0]

    def msearch(self, queries: np.ndarray, cutoff: int = 100):
        scores, ids = self.faiss_index.search(queries, cutoff)
        return ids, scores
