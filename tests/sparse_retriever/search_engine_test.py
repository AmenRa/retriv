from math import isclose

import pytest

from retriv import SearchEngine

REL_TOL = 1e-6


# FIXTURES =====================================================================
@pytest.fixture
def collection():
    return [
        {"id": 1, "text": "Shane"},
        {"id": 2, "text": "Shane C"},
        {"id": 3, "text": "Shane P Connelly"},
        {"id": 4, "text": "Shane Connelly"},
        {"id": 5, "text": "Shane Shane Connelly Connelly"},
        {"id": 6, "text": "Shane Shane Shane Connelly Connelly Connelly"},
    ]


def test_search_bm25(collection):
    se = SearchEngine(hyperparams=dict(b=0.5, k1=0))
    se.index(collection)

    query = "shane"

    results = se.search(query=query, return_docs=False)

    print(se.inverted_index)

    print(results)
    assert isclose(results[1], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[2], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[3], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[4], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[5], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[6], 0.07410797, rel_tol=REL_TOL)

    se.hyperparams = dict(b=0, k1=10)
    results = se.search(query=query, return_docs=False)
    print(results)
    assert isclose(results[1], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[2], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[3], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[4], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[5], 0.13586462, rel_tol=REL_TOL)
    assert isclose(results[6], 0.18812023, rel_tol=REL_TOL)

    se.hyperparams = dict(b=1, k1=5)
    results = se.search(query=query, return_docs=False)
    print(results)
    assert isclose(results[1], 0.16674294, rel_tol=REL_TOL)
    assert isclose(results[2], 0.10261103, rel_tol=REL_TOL)
    assert isclose(results[3], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results[4], 0.10261103, rel_tol=REL_TOL)
    assert isclose(results[5], 0.10261103, rel_tol=REL_TOL)
    assert isclose(results[6], 0.10261105, rel_tol=REL_TOL)


def test_msearch_bm25(collection):
    se = SearchEngine(hyperparams=dict(b=0.5, k1=0))
    se.index(collection)

    queries = [
        {"id": "q_1", "text": "shane"},
        {"id": "q_2", "text": "connelly"},
    ]

    results = se.msearch(queries=queries)

    print(results)
    assert isclose(results["q_1"][1], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][2], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][3], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][4], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][5], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][6], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_2"][3], 0.44183275, rel_tol=REL_TOL)
    assert isclose(results["q_2"][4], 0.44183275, rel_tol=REL_TOL)
    assert isclose(results["q_2"][5], 0.44183275, rel_tol=REL_TOL)
    assert isclose(results["q_2"][6], 0.44183275, rel_tol=REL_TOL)

    se.hyperparams = dict(b=0, k1=10)
    results = se.msearch(queries=queries)
    print(results)
    assert isclose(results["q_1"][1], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][2], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][3], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][4], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][5], 0.13586462, rel_tol=REL_TOL)
    assert isclose(results["q_1"][6], 0.18812023, rel_tol=REL_TOL)
    assert isclose(results["q_2"][3], 0.44183275, rel_tol=REL_TOL)
    assert isclose(results["q_2"][4], 0.44183275, rel_tol=REL_TOL)
    assert isclose(results["q_2"][5], 0.8100267, rel_tol=REL_TOL)
    assert isclose(results["q_2"][6], 1.1215755, rel_tol=REL_TOL)

    se.hyperparams = dict(b=1, k1=5)
    results = se.msearch(queries=queries)
    print(results)
    assert isclose(results["q_1"][1], 0.16674294, rel_tol=REL_TOL)
    assert isclose(results["q_1"][2], 0.10261103, rel_tol=REL_TOL)
    assert isclose(results["q_1"][3], 0.07410797, rel_tol=REL_TOL)
    assert isclose(results["q_1"][4], 0.10261103, rel_tol=REL_TOL)
    assert isclose(results["q_1"][5], 0.10261103, rel_tol=REL_TOL)
    assert isclose(results["q_1"][6], 0.10261105, rel_tol=REL_TOL)
    assert isclose(results["q_2"][3], 0.44183275, rel_tol=REL_TOL)
    assert isclose(results["q_2"][4], 0.6117684, rel_tol=REL_TOL)
    assert isclose(results["q_2"][5], 0.6117684, rel_tol=REL_TOL)
    assert isclose(results["q_2"][6], 0.6117684, rel_tol=REL_TOL)
