from ranx import Qrels, Run, evaluate, fuse, optimize_fusion


def tune_merger(qrels, runs, metric):
    ranx_qrels = Qrels(qrels)
    ranx_runs = [Run(run) for run in runs]

    best_score = 0.0
    best_config = None

    for norm in ["min-max", "max", "sum"]:
        best_params = optimize_fusion(
            qrels=ranx_qrels,
            runs=ranx_runs,
            norm=norm,
            method="wsum",
            metric=metric,
            show_progress=False,
        )

        combined_run = fuse(
            runs=ranx_runs, norm=norm, method="wsum", params=best_params
        )

        score = evaluate(ranx_qrels, combined_run, metric)
        if score > best_score:
            best_score = score
            best_config = {
                "norm": norm,
                "params": best_params,
            }

    return best_config
