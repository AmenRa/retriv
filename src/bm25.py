from typing import Tuple

import numba as nb
import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList


@njit(cache=True)
def concat1d(X):
    out = np.empty(sum([len(x) for x in X]), dtype=X[0].dtype)

    i = 0
    for x in X:
        for j in range(len(x)):
            out[i] = x[j]
            i = i + 1

    return out


@njit(cache=True)
def bm25(
    b: float,
    k1: float,
    term_doc_freqs: nb.typed.List[np.ndarray],
    doc_ids: nb.typed.List[np.ndarray],
    relative_doc_lens: nb.typed.List[np.ndarray],
    n_res: int,
) -> Tuple[np.ndarray]:
    unique_doc_ids = np.unique(concat1d(doc_ids))

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
    indices = np.argsort(scores)[::-1][:n_res]

    return unique_doc_ids[indices], scores[indices]


@njit(parallel=True, cache=True)
def bm25_multi(
    b: float,
    k1: float,
    term_doc_freqs: nb.typed.List[nb.typed.List[np.ndarray]],
    doc_ids: nb.typed.List[nb.typed.List[np.ndarray]],
    relative_doc_lens: nb.typed.List[nb.typed.List[np.ndarray]],
    n_res: int,
) -> Tuple[nb.typed.List[np.ndarray]]:
    unique_doc_ids = TypedList([np.empty(1, dtype=np.int32) for _ in doc_ids])
    scores = TypedList([np.empty(1, dtype=np.float32) for _ in doc_ids])

    for i in prange(len(term_doc_freqs)):
        unique_doc_ids[i], scores[i] = bm25(
            b=b,
            k1=k1,
            term_doc_freqs=term_doc_freqs[i],
            doc_ids=doc_ids[i],
            relative_doc_lens=relative_doc_lens,
            n_res=n_res,
        )

    return unique_doc_ids, scores
