import numpy as np
from numba import njit


# UNION ------------------------------------------------------------------------
@njit(cache=True)
def union_sorted(a1: np.array, a2: np.array):
    result = np.empty(len(a1) + len(a2), dtype=np.int32)
    i = 0
    j = 0
    k = 0

    while i < len(a1) and j < len(a2):
        if a1[i] < a2[j]:
            result[k] = a1[i]
            i += 1
        elif a1[i] > a2[j]:
            result[k] = a2[j]
            j += 1
        else:  # a1[i] == a2[j]
            result[k] = a1[i]
            i += 1
            j += 1
        k += 1

    result = result[:k]

    if i < len(a1):
        result = np.concatenate((result, a1[i:]))
    elif j < len(a2):
        result = np.concatenate((result, a2[j:]))

    return result


@njit(cache=True)
def union_sorted_multi(arrays):
    if len(arrays) == 1:
        return arrays[0]
    elif len(arrays) == 2:
        return union_sorted(arrays[0], arrays[1])
    else:
        return union_sorted(
            union_sorted_multi(arrays[:2]), union_sorted_multi(arrays[2:])
        )


# INTERSECTION -----------------------------------------------------------------
@njit(cache=True)
def intersect_sorted(a1: np.array, a2: np.array):
    result = np.empty(min(len(a1), len(a2)), dtype=np.int32)
    i = 0
    j = 0
    k = 0

    while i < len(a1) and j < len(a2):
        if a1[i] < a2[j]:
            i += 1
        elif a1[i] > a2[j]:
            j += 1
        else:  # a1[i] == a2[j]
            result[k] = a1[i]
            i += 1
            j += 1
            k += 1

    return result[:k]


@njit(cache=True)
def intersect_sorted_multi(arrays):
    a = arrays[0]

    for i in range(1, len(arrays)):
        a = intersect_sorted(a, arrays[i])

    return a


# DIFFERENCE -------------------------------------------------------------------
@njit(cache=True)
def diff_sorted(a1: np.array, a2: np.array):
    result = np.empty(len(a1), dtype=np.int32)
    i = 0
    j = 0
    k = 0

    while i < len(a1) and j < len(a2):
        if a1[i] < a2[j]:
            result[k] = a1[i]
            i += 1
            k += 1
        elif a1[i] > a2[j]:
            j += 1
        else:  # a1[i] == a2[j]
            i += 1
            j += 1

    result = result[:k]

    if i < len(a1):
        result = np.concatenate((result, a1[i:]))

    return result


#  -----------------------------------------------------------------------------
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
def get_indices(array, scores):
    n_scores = len(scores)
    min_score = min(scores)
    max_score = max(scores)
    indices = np.full(n_scores, -1, dtype=np.int64)
    counter = 0

    for i in range(len(array)):
        if array[i] >= min_score and array[i] <= max_score:
            for j in range(len(scores)):
                if indices[j] == -1:
                    if scores[j] == array[i]:
                        indices[j] = i
                        counter += 1
                        if len(indices) == counter:
                            return indices
                        break

    return indices


@njit(cache=True)
def unsorted_top_k(array: np.ndarray, k: int):
    top_k_values = np.zeros(k, dtype=np.float32)
    top_k_indices = np.zeros(k, dtype=np.int32)

    min_value = 0.0
    min_value_idx = 0

    for i, value in enumerate(array):
        if value > min_value:
            top_k_values[min_value_idx] = value
            top_k_indices[min_value_idx] = i
            min_value_idx = top_k_values.argmin()
            min_value = top_k_values[min_value_idx]

    return top_k_values, top_k_indices
