import numpy as np
from numba import njit


@njit(cache=True)
def join_sorted(a1, a2):
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
def join_sorted_multi(arrays):
    a = arrays[0]

    for i in range(1, len(arrays)):
        a = join_sorted(a, arrays[i])

    return a


@njit(cache=True)
def join_sorted_multi_recursive(arrays):
    if len(arrays) == 1:
        return arrays[0]
    elif len(arrays) == 2:
        return join_sorted(arrays[0], arrays[1])
    else:
        return join_sorted(
            join_sorted_multi(arrays[:2]), join_sorted_multi(arrays[2:])
        )


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
