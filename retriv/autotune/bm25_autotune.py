import warnings

import optuna
from optuna.exceptions import ExperimentalWarning
from ranx import Qrels, Run, evaluate

warnings.filterwarnings("ignore", category=ExperimentalWarning)


def bm25_objective(trial, queries, qrels, se, metric, cutoff):
    b = trial.suggest_float("b", 0.0, 1.0, step=0.01)
    k1 = trial.suggest_float("k1", 0.0, 10.0, step=0.1)

    se.hyperparams = dict(b=b, k1=k1)
    run = Run(se.bsearch(queries=queries, cutoff=cutoff, show_progress=False))

    return evaluate(qrels, run, metric)


def tune_bm25(queries, qrels, se, metric, n_trials, cutoff):
    qrels = Qrels(qrels)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: bm25_objective(trial, queries, qrels, se, metric, cutoff),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Set best params
    se.hyperparams = study.best_params

    return study.best_params
