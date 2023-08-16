from typing import Dict, List


def extract_scores(results):
    """Extract the scores from a given results dictionary."""
    scores = [None] * len(results)
    for i, v in enumerate(results.values()):
        scores[i] = v
    return scores


def safe_max(x):
    return max(x) if len(x) != 0 else 0


def safe_min(x):
    return min(x) if len(x) != 0 else 0


def min_max_norm(run: Dict[str, float]):
    """Apply `min-max normalization` to a given run."""
    normalized_run = {}

    for q_id, results in run.items():
        scores = extract_scores(results)
        min_score = safe_min(scores)
        max_score = safe_max(scores)
        denominator = max(max_score - min_score, 1e-9)

        normalized_results = {
            doc_id: (results[doc_id] - min_score) / (denominator) for doc_id in results
        }

        normalized_run[q_id] = normalized_results

    return normalized_run


def max_norm(run: Dict[str, float]):
    """Apply `max normalization` to a given run."""
    normalized_run = {}

    for q_id, results in run.items():
        scores = extract_scores(results)
        max_score = safe_max(scores)
        denominator = max(max_score, 1e-9)

        normalized_results = {
            doc_id: results[doc_id] / denominator for doc_id in results
        }

        normalized_run[q_id] = normalized_results

    return normalized_run


def sum_norm(run: Dict[str, float]):
    """Apply `sum normalization` to a given run."""
    normalized_run = {}

    for q_id, results in run.items():
        scores = extract_scores(results)
        min_score = safe_min(scores)
        sum_score = sum(scores)
        denominator = sum_score - min_score * len(results)
        denominator = max(denominator, 1e-9)

        normalized_results = {
            doc_id: (results[doc_id] - min_score) / (denominator) for doc_id in results
        }

        normalized_run[q_id] = normalized_results

    return normalized_run


def min_max_norm_multi(runs: List[Dict[str, float]]):
    """Apply `min-max normalization` to a list of given runs."""
    return [min_max_norm(run) for run in runs]


def max_norm_multi(runs: List[Dict[str, float]]):
    """Apply `max normalization` to a list of given runs."""
    return [max_norm(run) for run in runs]


def sum_norm_multi(runs: List[Dict[str, float]]):
    """Apply `sum normalization` to a list of given runs."""
    return [sum_norm(run) for run in runs]
