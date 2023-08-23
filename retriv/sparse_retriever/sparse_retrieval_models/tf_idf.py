from typing import Tuple

import numba as nb
import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ...utils.numba_utils import (
    intersect_sorted,
    intersect_sorted_multi,
    union_sorted_multi,
    unsorted_top_k,
)


@njit(cache=True)
def tf_idf(
    term_doc_freqs: nb.typed.List[np.ndarray],
    doc_ids: nb.typed.List[np.ndarray],
    doc_lens: nb.typed.List[np.ndarray],
    cutoff: int,
    operator: str = "OR",
    subset_doc_ids: np.ndarray = None,
) -> Tuple[np.ndarray]:
    if operator == "AND":
        unique_doc_ids = intersect_sorted_multi(doc_ids)
    elif operator == "OR":
        unique_doc_ids = union_sorted_multi(doc_ids)

    if subset_doc_ids is not None:
        unique_doc_ids = intersect_sorted(unique_doc_ids, subset_doc_ids)

    doc_count = len(doc_lens)
    scores = np.empty(doc_count, dtype=np.float32)
    scores[unique_doc_ids] = 0.0  # Initialize scores

    for i in range(len(term_doc_freqs)):
        indices = doc_ids[i]
        freqs = term_doc_freqs[i]

        tf = freqs / doc_lens[indices]

        df = np.float32(len(indices))
        idf = np.float32(np.log((1.0 + doc_count) / (1.0 + df)) + 1.0)

        scores[indices] += tf * idf

    scores = scores[unique_doc_ids]

    if cutoff < len(scores):
        scores, indices = unsorted_top_k(scores, cutoff)
        unique_doc_ids = unique_doc_ids[indices]

    indices = np.argsort(-scores)

    return unique_doc_ids[indices], scores[indices]


@njit(cache=True, parallel=True)
def tf_idf_multi(
    term_doc_freqs: nb.typed.List[nb.typed.List[np.ndarray]],
    doc_ids: nb.typed.List[nb.typed.List[np.ndarray]],
    doc_lens: nb.typed.List[nb.typed.List[np.ndarray]],
    cutoff: int,
) -> Tuple[nb.typed.List[np.ndarray]]:
    unique_doc_ids = TypedList([np.empty(1, dtype=np.int32) for _ in doc_ids])
    scores = TypedList([np.empty(1, dtype=np.float32) for _ in doc_ids])

    for i in prange(len(term_doc_freqs)):
        _term_doc_freqs = term_doc_freqs[i]
        _doc_ids = doc_ids[i]

        _unique_doc_ids = union_sorted_multi(_doc_ids)

        doc_count = len(doc_lens)
        _scores = np.empty(doc_count, dtype=np.float32)
        _scores[_unique_doc_ids] = 0.0  # Initialize _scores

        for j in range(len(_term_doc_freqs)):
            indices = _doc_ids[j]
            freqs = _term_doc_freqs[j]

            tf = freqs / doc_lens[indices]

            df = np.float32(len(indices))
            idf = np.float32(np.log((1.0 + doc_count) / (1.0 + df)) + 1.0)

            _scores[indices] += tf * idf

        _scores = _scores[_unique_doc_ids]

        if cutoff < len(_scores):
            _scores, indices = unsorted_top_k(_scores, cutoff)
            _unique_doc_ids = _unique_doc_ids[indices]

        indices = np.argsort(_scores)[::-1]

        unique_doc_ids[i] = _unique_doc_ids[indices]
        scores[i] = _scores[indices]

    return unique_doc_ids, scores
