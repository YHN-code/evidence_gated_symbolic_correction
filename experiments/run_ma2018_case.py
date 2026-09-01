from __future__ import annotations

import argparse

import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from asrc.agent.policy import run_agentic_search
from asrc.data.loaders import load_case
from asrc.evaluation.metrics import regression_metrics
from asrc.utils.io import ensure_run_dir, write_json, write_table_bundle
from asrc.utils.progress import mark_done, progress_message, should_skip


def run(
    config: str,
    run_id: str,
    resume: bool = False,
    force: bool = False,
    verbose: bool = False,
    llm: bool = False,
    allow_llm_fallback: bool = False,
    llm_config: str | None = None,
) -> None:
    run_dir = ensure_run_dir(run_id)
    expected = [run_dir / "metrics" / "ma2018_metrics.csv", run_dir / "data" / "ma2018_predictions.csv"]
    if should_skip(run_dir, "ma2018_case", expected, resume, force):
        if verbose:
            print("Skipping ma2018_case; completed outputs found.")
        return

    case_all = load_case(config)
    progress_message(run_dir, "Ma 2018 case started", verbose, config=config)
    group_col = case_all.group or "dataset"
    alpha_grid = case_all.config["symbolic"]["alpha_grid"]
    max_terms = case_all.config["symbolic"]["max_terms"]["asrc"]
    prior_level = case_all.config.get("prior_level", "mechanics_informed")
    metric_rows = []
    formula_rows = []
    prediction_rows = []
    correction_method = "llm_iterative_asrc" if llm else "asrc"

    for dataset in sorted(case_all.frame[group_col].unique().tolist()):
        progress_message(run_dir, "Ma 2018 subgroup started", verbose, dataset=dataset)
        case = case_all.subset(**{group_col: dataset})
        agent_result = run_agentic_search(
            case,
            run_dir,
            f"ma2018_{dataset}",
            mode="asrc",
            max_terms=max_terms,
            alpha_grid=alpha_grid,
            enforce_physical=True,
            use_llm=llm,
            allow_fallback=allow_llm_fallback,
            llm_config=llm_config,
            metadata={"case": "ma2018", "rock": dataset, "dataset": dataset, "prior_level": prior_level},
            verbose=verbose,
        )
        baseline = regression_metrics(case.frame[case.y_true].to_numpy(float), case.frame[case.y_base].to_numpy(float)).to_dict()
        result = agent_result.result
        metric_rows.append({"case": "ma2018", "rock": dataset, "prior_level": prior_level, "method": "baseline", "predictive_status": "baseline", "structural_status": "not_applicable", "formula": "0", "complexity": 0, "physical_violations": 0, **baseline})
        metric_rows.append(
            {
                "case": "ma2018",
                "rock": dataset,
                "prior_level": prior_level,
                "method": correction_method,
                "result_status": agent_result.result_status,
                "predictive_status": agent_result.predictive_status,
                "structural_status": agent_result.structural_status,
                "acceptance_accepted": agent_result.result_status == "accepted" if llm else True,
                "acceptance_blockers": ";".join(agent_result.final_audit.get("blockers", [])),
                "formula": result.best["formula"],
                "complexity": result.best["complexity"],
                "physical_violations": result.best["physical_violations"],
                "leave_one_beta_rmse": result.best["leave_one_beta_rmse"],
                "selection_validation_rmse": result.best.get("selection_validation_rmse", float("nan")),
                "selection_metric": result.best.get("selection_metric", "training_rmse"),
                **{key: result.best[key] for key in ["rmse", "mae", "mape", "r2", "max_abs_error"]},
            }
        )
        formula_rows.append(
            {
                "dataset": dataset,
                "prior_level": prior_level,
                "method": correction_method,
                "result_status": agent_result.result_status,
                "predictive_status": agent_result.predictive_status,
                "structural_status": agent_result.structural_status,
                "acceptance_blockers": agent_result.final_audit.get("blockers", []),
                **result.best,
            }
        )
        prediction_rows.append(result.predictions)
        agent_result.candidates.to_csv(run_dir / "formulas" / f"ma2018_{dataset}_candidate_formulas.csv", index=False)
        progress_message(run_dir, "Ma 2018 subgroup finished", verbose, dataset=dataset, rmse=f"{result.best['rmse']:.6g}")

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(run_dir / "data" / "ma2018_predictions.csv", index=False)
    write_json(run_dir / "formulas" / "ma2018_best_formulas.json", formula_rows)
    write_table_bundle(metrics, run_dir / "metrics" / "ma2018_metrics")
    mark_done(run_dir, "ma2018_case", {"rows": len(predictions)})
    progress_message(run_dir, "Ma 2018 case finished", verbose, rows=len(predictions))
    if verbose:
        print(f"Ma 2018 case complete: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ma2018_case.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--allow-llm-fallback", action="store_true")
    parser.add_argument("--llm-config", default=None)
    args = parser.parse_args()
    run(args.config, args.run_id, args.resume, args.force, args.verbose, args.llm, args.allow_llm_fallback, args.llm_config)


if __name__ == "__main__":
    main()
