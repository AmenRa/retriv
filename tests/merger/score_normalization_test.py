from math import isclose

import pytest

from retriv.merger.normalization import (
    extract_scores,
    max_norm,
    max_norm_multi,
    min_max_norm,
    min_max_norm_multi,
    safe_max,
    safe_min,
    sum_norm,
    sum_norm_multi,
)


# FIXTURES =====================================================================
@pytest.fixture
def run_a():
    return {
        "q1": {
            "d1": 2.0,
            "d2": 0.7,
            "d3": 0.5,
        },
        "q2": {
            "d1": 1.0,
            "d2": 0.7,
            "d3": 0.5,
        },
    }


@pytest.fixture
def run_b():
    return {
        "q1": {
            "d3": 2.0,
            "d1": 0.7,
        },
        "q2": {
            "d1": 1.0,
            "d2": 0.7,
            "d3": 0.5,
        },
    }


# TESTS ========================================================================
def test_extract_scores(run_a, run_b):
    assert extract_scores(run_a["q1"]) == [2.0, 0.7, 0.5]
    assert extract_scores(run_a["q2"]) == [1.0, 0.7, 0.5]
    assert extract_scores(run_b["q1"]) == [2.0, 0.7]
    assert extract_scores(run_b["q2"]) == [1.0, 0.7, 0.5]


def test_safe_max():
    assert safe_max([1.0, 0.7, 0.5]) == 1.0
    assert safe_max([]) == 0.0


def test_safe_min():
    assert safe_min([1.0, 0.7, 0.5]) == 0.5
    assert safe_min([]) == 0.0


def test_max_norm(run_a, run_b):
    normalized_run = max_norm(run_a)

    assert isclose(normalized_run["q1"]["d1"], 1.0)
    assert isclose(normalized_run["q1"]["d2"], 0.7 / 2.0)
    assert isclose(normalized_run["q1"]["d3"], 0.5 / 2.0)

    assert isclose(normalized_run["q2"]["d1"], 1.0)
    assert isclose(normalized_run["q2"]["d2"], 0.7)
    assert isclose(normalized_run["q2"]["d3"], 0.5)

    normalized_run = max_norm(run_b)

    assert isclose(normalized_run["q1"]["d3"], 1.0)
    assert isclose(normalized_run["q1"]["d1"], 0.7 / 2.0)

    assert isclose(normalized_run["q2"]["d1"], 1.0)
    assert isclose(normalized_run["q2"]["d2"], 0.7)
    assert isclose(normalized_run["q2"]["d3"], 0.5)


def test_min_max_norm(run_a, run_b):
    normalized_run = min_max_norm(run_a)

    assert isclose(normalized_run["q1"]["d1"], (2.0 - 0.5) / (2.0 - 0.5))
    assert isclose(normalized_run["q1"]["d2"], (0.7 - 0.5) / (2.0 - 0.5))
    assert isclose(normalized_run["q1"]["d3"], (0.5 - 0.5) / (2.0 - 0.5))

    assert isclose(normalized_run["q2"]["d1"], (1.0 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_run["q2"]["d2"], (0.7 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_run["q2"]["d3"], (0.5 - 0.5) / (1.0 - 0.5))

    normalized_run = min_max_norm(run_b)

    assert isclose(normalized_run["q1"]["d3"], (2.0 - 0.7) / (2.0 - 0.7))
    assert isclose(normalized_run["q1"]["d1"], (0.7 - 0.7) / (2.0 - 0.7))

    assert isclose(normalized_run["q2"]["d1"], (1.0 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_run["q2"]["d2"], (0.7 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_run["q2"]["d3"], (0.5 - 0.5) / (1.0 - 0.5))


def test_sum_norm(run_a, run_b):
    normalized_run = sum_norm(run_a)

    denominator = (2.0 - 0.5) + (0.7 - 0.5) + (0.5 - 0.5)
    assert isclose(normalized_run["q1"]["d1"], (2.0 - 0.5) / denominator)
    assert isclose(normalized_run["q1"]["d2"], (0.7 - 0.5) / denominator)
    assert isclose(normalized_run["q1"]["d3"], (0.5 - 0.5) / denominator)

    denominator = (1.0 - 0.5) + (0.7 - 0.5) + (0.5 - 0.5)
    assert isclose(normalized_run["q2"]["d1"], (1.0 - 0.5) / denominator)
    assert isclose(normalized_run["q2"]["d2"], (0.7 - 0.5) / denominator)
    assert isclose(normalized_run["q2"]["d3"], (0.5 - 0.5) / denominator)

    normalized_run = sum_norm(run_b)

    denominator = (1.0 - 0.7) + (0.7 - 0.7)
    assert isclose(normalized_run["q1"]["d3"], (1.0 - 0.7) / denominator)
    assert isclose(normalized_run["q1"]["d1"], (0.7 - 0.7) / denominator)

    denominator = (1.0 - 0.5) + (0.7 - 0.5) + (0.5 - 0.5)
    assert isclose(normalized_run["q2"]["d1"], (1.0 - 0.5) / denominator)
    assert isclose(normalized_run["q2"]["d2"], (0.7 - 0.5) / denominator)
    assert isclose(normalized_run["q2"]["d3"], (0.5 - 0.5) / denominator)


def test_max_norm_multi(run_a, run_b):
    runs = [run_a, run_b]
    normalized_runs = max_norm_multi(runs)

    assert isclose(normalized_runs[0]["q1"]["d1"], 1.0)
    assert isclose(normalized_runs[0]["q1"]["d2"], 0.7 / 2.0)
    assert isclose(normalized_runs[0]["q1"]["d3"], 0.5 / 2.0)
    assert isclose(normalized_runs[0]["q2"]["d1"], 1.0)
    assert isclose(normalized_runs[0]["q2"]["d2"], 0.7)
    assert isclose(normalized_runs[0]["q2"]["d3"], 0.5)

    assert isclose(normalized_runs[1]["q1"]["d3"], 1.0)
    assert isclose(normalized_runs[1]["q1"]["d1"], 0.7 / 2.0)
    assert isclose(normalized_runs[1]["q2"]["d1"], 1.0)
    assert isclose(normalized_runs[1]["q2"]["d2"], 0.7)
    assert isclose(normalized_runs[1]["q2"]["d3"], 0.5)


def test_min_max_norm_multi(run_a, run_b):
    runs = [run_a, run_b]
    normalized_runs = min_max_norm_multi(runs)

    assert isclose(normalized_runs[0]["q1"]["d1"], (2.0 - 0.5) / (2.0 - 0.5))
    assert isclose(normalized_runs[0]["q1"]["d2"], (0.7 - 0.5) / (2.0 - 0.5))
    assert isclose(normalized_runs[0]["q1"]["d3"], (0.5 - 0.5) / (2.0 - 0.5))
    assert isclose(normalized_runs[0]["q2"]["d1"], (1.0 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_runs[0]["q2"]["d2"], (0.7 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_runs[0]["q2"]["d3"], (0.5 - 0.5) / (1.0 - 0.5))

    assert isclose(normalized_runs[1]["q1"]["d3"], (2.0 - 0.7) / (2.0 - 0.7))
    assert isclose(normalized_runs[1]["q1"]["d1"], (0.7 - 0.7) / (2.0 - 0.7))
    assert isclose(normalized_runs[1]["q2"]["d1"], (1.0 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_runs[1]["q2"]["d2"], (0.7 - 0.5) / (1.0 - 0.5))
    assert isclose(normalized_runs[1]["q2"]["d3"], (0.5 - 0.5) / (1.0 - 0.5))


def test_sum_norm_multi(run_a, run_b):
    runs = [run_a, run_b]
    normalized_runs = sum_norm_multi(runs)

    denominator = (2.0 - 0.5) + (0.7 - 0.5) + (0.5 - 0.5)
    assert isclose(normalized_runs[0]["q1"]["d1"], (2.0 - 0.5) / denominator)
    assert isclose(normalized_runs[0]["q1"]["d2"], (0.7 - 0.5) / denominator)
    assert isclose(normalized_runs[0]["q1"]["d3"], (0.5 - 0.5) / denominator)

    denominator = (1.0 - 0.5) + (0.7 - 0.5) + (0.5 - 0.5)
    assert isclose(normalized_runs[0]["q2"]["d1"], (1.0 - 0.5) / denominator)
    assert isclose(normalized_runs[0]["q2"]["d2"], (0.7 - 0.5) / denominator)
    assert isclose(normalized_runs[0]["q2"]["d3"], (0.5 - 0.5) / denominator)

    denominator = (1.0 - 0.7) + (0.7 - 0.7)
    assert isclose(normalized_runs[1]["q1"]["d3"], (1.0 - 0.7) / denominator)
    assert isclose(normalized_runs[1]["q1"]["d1"], (0.7 - 0.7) / denominator)
    denominator = (1.0 - 0.5) + (0.7 - 0.5) + (0.5 - 0.5)
    assert isclose(normalized_runs[1]["q2"]["d1"], (1.0 - 0.5) / denominator)
    assert isclose(normalized_runs[1]["q2"]["d2"], (0.7 - 0.5) / denominator)
    assert isclose(normalized_runs[1]["q2"]["d3"], (0.5 - 0.5) / denominator)
