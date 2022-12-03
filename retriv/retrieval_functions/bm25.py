from typing import Tuple

import numba as nb
import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ..utils.numba_utils import get_indices, join_sorted_multi_recursive


@njit(cache=True)
def bm25(
    b: float,
    k1: float,
    term_doc_freqs: nb.typed.List[np.ndarray],
    doc_ids: nb.typed.List[np.ndarray],
    relative_doc_lens: nb.typed.List[np.ndarray],
    cutoff: int,
) -> Tuple[np.ndarray]:
    unique_doc_ids = join_sorted_multi_recursive(doc_ids)

    doc_count = len(relative_doc_lens)
    scores = np.empty(doc_count, dtype=np.float32)
    scores[unique_doc_ids] = 0.0  # Initialize scores

    for i in range(len(term_doc_freqs)):
        indices = doc_ids[i]
        freqs = term_doc_freqs[i]

        df = np.float32(len(indices))
        idf = np.float32(np.log(1.0 + (((doc_count - df) + 0.5) / (df + 0.5))))

        # BM25_score = IDF * (tf * (k + 1)) / (k * (1.0 - b + b * (|d|/avgDl)) + tf)
        scores[indices] += idf * (
            (freqs * (k1 + 1.0))
            / (freqs + k1 * (1.0 - b + (b * relative_doc_lens[indices])))
        )

    scores = scores[unique_doc_ids]

    if cutoff < len(scores):
        top_scores = -np.partition(-scores, cutoff)[:cutoff]
        indices = get_indices(scores, top_scores)
        unique_doc_ids = unique_doc_ids[indices]
        scores = scores[indices]

    indices = np.argsort(scores)[::-1]

    return unique_doc_ids[indices], scores[indices]


# @njit(parallel=True, cache=True)
# def bm25_multi(
#     b: float,
#     k1: float,
#     term_doc_freqs: nb.typed.List[nb.typed.List[np.ndarray]],
#     doc_ids: nb.typed.List[nb.typed.List[np.ndarray]],
#     relative_doc_lens: nb.typed.List[nb.typed.List[np.ndarray]],
#     cutoff: int,
# ) -> Tuple[nb.typed.List[np.ndarray]]:
#     unique_doc_ids = TypedList([np.empty(1, dtype=np.int32) for _ in doc_ids])
#     scores = TypedList([np.empty(1, dtype=np.float32) for _ in doc_ids])

#     for i in prange(len(term_doc_freqs)):
#         unique_doc_ids[i], scores[i] = bm25(
#             b=b,
#             k1=k1,
#             term_doc_freqs=term_doc_freqs[i],
#             doc_ids=doc_ids[i],
#             relative_doc_lens=relative_doc_lens,
#             cutoff=cutoff,
#         )

#     return unique_doc_ids, scores


@njit(parallel=True, cache=True)
def bm25_multi(
    b: float,
    k1: float,
    term_doc_freqs: nb.typed.List[nb.typed.List[np.ndarray]],
    doc_ids: nb.typed.List[nb.typed.List[np.ndarray]],
    relative_doc_lens: nb.typed.List[nb.typed.List[np.ndarray]],
    cutoff: int,
) -> Tuple[nb.typed.List[np.ndarray]]:
    unique_doc_ids = TypedList([np.empty(1, dtype=np.int32) for _ in doc_ids])
    scores = TypedList([np.empty(1, dtype=np.float32) for _ in doc_ids])

    for i in prange(len(term_doc_freqs)):
        _term_doc_freqs = term_doc_freqs[i]
        _doc_ids = doc_ids[i]

        _unique_doc_ids = join_sorted_multi_recursive(_doc_ids)

        doc_count = len(relative_doc_lens)
        _scores = np.empty(doc_count, dtype=np.float32)
        _scores[_unique_doc_ids] = 0.0  # Initialize _scores

        for j in range(len(_term_doc_freqs)):
            indices = _doc_ids[j]
            freqs = _term_doc_freqs[j]

            df = np.float32(len(indices))
            idf = np.float32(
                np.log(1.0 + (((doc_count - df) + 0.5) / (df + 0.5)))
            )

            # BM25_score = IDF * (tf * (k + 1)) / (k * (1.0 - b + b * (|d|/avgDl)) + tf)
            _scores[indices] += idf * (
                (freqs * (k1 + 1.0))
                / (freqs + k1 * (1.0 - b + (b * relative_doc_lens[indices])))
            )

        _scores = _scores[_unique_doc_ids]

        if cutoff < len(_scores):
            top_scores = -np.partition(-_scores, cutoff)[:cutoff]
            indices = get_indices(_scores, top_scores)
            _unique_doc_ids = _unique_doc_ids[indices]
            _scores = _scores[indices]

        indices = np.argsort(_scores)[::-1]

        unique_doc_ids[i] = _unique_doc_ids[indices]
        scores[i] = _scores[indices]

    return unique_doc_ids, scores
