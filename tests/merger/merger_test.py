from math import isclose

import pytest

from retriv.merger.merger import Merger
from retriv.merger.normalization import min_max_norm


# FIXTURES =====================================================================
@pytest.fixture
def merger():
    return Merger()


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
def test_fuse(merger, run_a, run_b):
    fused_results = merger.fuse([run_a["q1"], run_b["q1"]])

    norm_run_a = min_max_norm(run_a)
    norm_run_b = min_max_norm(run_b)

    assert isclose(
        fused_results["d1"], norm_run_a["q1"]["d1"] + norm_run_b["q1"]["d1"]
    )
    assert isclose(fused_results["d2"], norm_run_a["q1"]["d2"])
    assert isclose(
        fused_results["d3"], norm_run_a["q1"]["d3"] + norm_run_b["q1"]["d3"]
    )


def test_mfuse(merger, run_a, run_b):
    fused_run = merger.mfuse([run_a, run_b])

    norm_run_a = min_max_norm(run_a)
    norm_run_b = min_max_norm(run_b)

    assert isclose(
        fused_run["q1"]["d1"], norm_run_a["q1"]["d1"] + norm_run_b["q1"]["d1"]
    )
    assert isclose(fused_run["q1"]["d2"], norm_run_a["q1"]["d2"])
    assert isclose(
        fused_run["q1"]["d3"], norm_run_a["q1"]["d3"] + norm_run_b["q1"]["d3"]
    )

    assert isclose(
        fused_run["q2"]["d1"], norm_run_a["q2"]["d1"] + norm_run_b["q2"]["d1"]
    )
    assert isclose(
        fused_run["q2"]["d2"], norm_run_a["q2"]["d2"] + norm_run_b["q2"]["d2"]
    )
    assert isclose(
        fused_run["q2"]["d3"], norm_run_a["q2"]["d3"] + norm_run_b["q2"]["d3"]
    )
