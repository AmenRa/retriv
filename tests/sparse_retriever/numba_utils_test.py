import numpy as np
import pytest
from numba.typed import List as TypedList

from retriv.utils.numba_utils import (
    concat1d,
    get_indices,
    join_sorted,
    join_sorted_multi,
    join_sorted_multi_recursive,
    unsorted_top_k,
)


# TESTS ========================================================================
def test_join_sorted():
    a1 = np.array([1, 3, 4, 7], dtype=np.int32)
    a2 = np.array([1, 4, 7, 9], dtype=np.int32)
    result = join_sorted(a1, a2)
    expected = np.array([1, 3, 4, 7, 9], dtype=np.int32)

    assert np.array_equal(result, expected)


def test_join_sorted_multi():
    a1 = np.array([1, 3, 4, 7], dtype=np.int32)
    a2 = np.array([1, 4, 7, 9], dtype=np.int32)
    a3 = np.array([10, 11], dtype=np.int32)
    a4 = np.array([11, 12, 13], dtype=np.int32)

    arrays = TypedList([a1, a2, a3, a4])

    result = join_sorted_multi(arrays)
    expected = np.array([1, 3, 4, 7, 9, 10, 11, 12, 13], dtype=np.int32)

    assert np.array_equal(result, expected)


def test_join_sorted_multi_recursive():
    a1 = np.array([1, 3, 4, 7], dtype=np.int32)
    a2 = np.array([1, 4, 7, 9], dtype=np.int32)
    a3 = np.array([10, 11], dtype=np.int32)
    a4 = np.array([11, 12, 13], dtype=np.int32)

    arrays = TypedList([a1, a2, a3, a4])

    result = join_sorted_multi_recursive(arrays)
    expected = np.array([1, 3, 4, 7, 9, 10, 11, 12, 13], dtype=np.int32)

    assert np.array_equal(result, expected)


def test_concat1d():
    a1 = np.array([1, 3, 4, 7], dtype=np.int32)
    a2 = np.array([1, 4, 7, 9], dtype=np.int32)
    a3 = np.array([10, 11], dtype=np.int32)
    a4 = np.array([11, 12, 13], dtype=np.int32)

    arrays = TypedList([a1, a2, a3, a4])

    result = concat1d(arrays)
    expected = np.array(
        [1, 3, 4, 7, 1, 4, 7, 9, 10, 11, 11, 12, 13], dtype=np.int32
    )

    assert np.array_equal(result, expected)


def test_get_indices():
    array = np.array([0.1, 0.3, 0.2, 0.4], dtype=np.float32)
    scores = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)

    result = get_indices(array, scores)
    expected = np.array([3, 1, 2, 0], dtype=np.int64)

    assert np.array_equal(result, expected)


def test_unsorted_top_k():
    array = np.array([0.1, 0.3, 0.2, 0.4], dtype=np.float32)
    k = 2

    top_k_values, top_k_indices = unsorted_top_k(array, k)

    assert len(top_k_values) == 2
    assert 0.3 in top_k_values
    assert 0.4 in top_k_values
    assert len(top_k_indices) == 2
    assert 1 in top_k_indices
    assert 3 in top_k_indices
