from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..autotune import tune_merger
from ..paths import merger_state_path
from .normalization import max_norm_multi, min_max_norm_multi, sum_norm_multi


class Merger:
    def __init__(self, index_name: str = "new-index"):
        self.index_name = index_name
        self.norm = "min-max"
        self.params = None

    def fuse(
        self, results: List[Dict[str, float]], cutoff: int = 100
    ) -> Dict[str, float]:
        return self.mfuse([{"q_0": res} for res in results], cutoff)["q_0"]

    def mfuse(
        self, runs: List[Dict[str, Dict[str, float]]], cutoff: int = 100
    ) -> Dict[str, Dict[str, float]]:
        if self.norm == "min-max":
            normalized_runs = min_max_norm_multi(runs)
        elif self.norm == "max":
            normalized_runs = max_norm_multi(runs)
        elif self.norm == "sum":
            normalized_runs = sum_norm_multi(runs)
        else:
            raise NotImplementedError

        weights = [1.0 for _ in runs] if self.params is None else self.params["weights"]

        fused_run = defaultdict(lambda: defaultdict(float))
        for i, run in enumerate(normalized_runs):
            for q_id in run:
                for doc_id in run[q_id]:
                    fused_run[q_id][doc_id] += weights[i] * run[q_id][doc_id]

        # Sort results by descending value and ascending key
        for q_id, results in list(fused_run.items()):
            fused_run[q_id] = dict(sorted(results.items(), key=lambda x: (-x[1], x[0])))

        # Apply cutoff
        for q_id, results in list(fused_run.items()):
            fused_run[q_id] = dict(list(results.items())[:cutoff])

        return dict(fused_run)

    def save(self):
        state = dict(
            init_args=dict(index_name=self.index_name),
            norm=self.norm,
            params=self.params,
        )
        np.savez_compressed(merger_state_path(self.index_name), state=state)

    @staticmethod
    def load(index_name: str = "new-index"):
        state = np.load(merger_state_path(index_name), allow_pickle=True)["state"][()]
        merger = Merger(**state["init_args"])
        merger.norm = state["norm"]
        merger.params = state["params"]
        return merger

    def autotune(
        self,
        qrels: Dict[str, Dict[str, float]],
        runs: List[Dict[str, Dict[str, float]]],
        metric: str = "ndcg",
    ):
        config = tune_merger(qrels, runs, metric)
        self.norm = config["norm"]
        self.params = config["params"]
        self.save()
