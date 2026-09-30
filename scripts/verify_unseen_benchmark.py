"""Verification and audit script for Phase 10 unseen benchmark experiments.

Checks all output artifacts, verifies trial counts, monotonicity, metrics, and prints comparative summaries.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd

from src.confidence.acquisition import HPO_SEARCH_SPACE
from src.data.loader import UNSEEN_DATASETS, get_project_root
from src.evaluation.unseen_benchmark import run_unseen_benchmark



def main():
    root = get_project_root()
    out_dir = root / "results" / "experiments" / "unseen_benchmark"
    trials_csv = out_dir / "unseen_trial_results.csv"
    summary_csv = out_dir / "unseen_benchmark_summary.csv"
    curves_csv = out_dir / "unseen_efficiency_curves.csv"
    metadata_json = out_dir / "unseen_benchmark_metadata.json"

    # 1. Run benchmark if artifacts don't exist yet
    if not (trials_csv.exists() and summary_csv.exists() and curves_csv.exists() and metadata_json.exists()):
        print("Benchmark artifacts not found. Running unseen benchmark now...")
        run_unseen_benchmark(
            dataset_names=list(UNSEEN_DATASETS),
            n_trials=20,
            seed=42,
            beta=0.1,
            output_dir=out_dir,
        )

    print("\n=======================================================")
    print("PHASE 10: UNSEEN DATASET BENCHMARK AUDIT & VERIFICATION")
    print("=======================================================\n")

    # Load artifacts
    df_trials = pd.read_csv(trials_csv)
    df_summary = pd.read_csv(summary_csv)
    df_curves = pd.read_csv(curves_csv)
    with open(metadata_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. Audits
    # Trial results row count
    expected_trials = len(UNSEEN_DATASETS) * 3 * 20  # 4 * 3 * 20 = 240
    assert len(df_trials) == expected_trials, f"Expected {expected_trials} trial rows, got {len(df_trials)}"
    print(f"[PASS] Trial results count: {len(df_trials)} rows (exactly {expected_trials}).")

    # Summary row count
    expected_summaries = len(UNSEEN_DATASETS) * 3  # 12
    assert len(df_summary) == expected_summaries, f"Expected {expected_summaries} summary rows, got {len(df_summary)}"
    print(f"[PASS] Benchmark summary count: {len(df_summary)} rows (exactly {expected_summaries}).")

    # Efficiency curves row count
    assert len(df_curves) == expected_trials, f"Expected {expected_trials} curve rows, got {len(df_curves)}"
    print(f"[PASS] Efficiency curve count: {len(df_curves)} rows (exactly {expected_trials}).")

    # Check columns
    required_trial_cols = [
        "dataset_name", "method", "trial_number", "cv_macro_f1", "best_so_far_cv_macro_f1", "duration_seconds"
    ] + list(HPO_SEARCH_SPACE.keys())
    for col in required_trial_cols:
        assert col in df_trials.columns, f"Missing column in trials: {col}"
    print("[PASS] All required hyperparameter and metric columns present in trial results.")

    # Check monotonic best_so_far for every dataset/method combination
    for (ds, m), group in df_trials.groupby(["dataset_name", "method"]):
        assert len(group) == 20, f"Expected 20 trials for {ds} - {m}, got {len(group)}"
        best_so_far_vals = group.sort_values("trial_number")["best_so_far_cv_macro_f1"].tolist()
        for i in range(len(best_so_far_vals) - 1):
            assert best_so_far_vals[i] <= best_so_far_vals[i + 1] + 1e-9, (
                f"Non-monotonic best_so_far in {ds} - {m} at trial {i}"
            )
    print("[PASS] best_so_far_cv_macro_f1 is strictly monotonically non-decreasing for all runs.")

    # Check hyperparameter bounds
    for p, (low, high, ptype) in HPO_SEARCH_SPACE.items():
        assert (df_trials[p] >= low - 1e-5).all(), f"Parameter {p} below lower bound {low}"
        assert (df_trials[p] <= high + 1e-5).all(), f"Parameter {p} above upper bound {high}"
    print("[PASS] All evaluated hyperparameters adhere strictly to search space bounds.")

    # Check metadata
    assert metadata["random_seed"] == 42
    assert metadata["n_trials_per_method"] == 20
    assert metadata["warm_start_count"] == 5
    assert metadata["acquisition_beta"] == 0.1
    assert set(metadata["datasets_evaluated"]) == set(UNSEEN_DATASETS)
    print("[PASS] Benchmark metadata correctly recorded.")

    # 3. Formatted Performance Reporting
    print("\n" + "=" * 80)
    print("TABLE 1: BEST VALIDATION CV MACRO-F1 (Mean across 3 folds)")
    print("=" * 80)
    pvt_cv = df_summary.pivot(index="dataset_name", columns="method", values="best_cv_macro_f1")
    pvt_cv = pvt_cv[["random_search", "optuna_tpe", "metahpo_c"]]
    print(pvt_cv.to_string())

    print("\n" + "=" * 80)
    print("TABLE 2: FINAL HELD-OUT TEST MACRO-F1 (Evaluated once on 20% test split)")
    print("=" * 80)
    pvt_test_f1 = df_summary.pivot(index="dataset_name", columns="method", values="test_macro_f1")
    pvt_test_f1 = pvt_test_f1[["random_search", "optuna_tpe", "metahpo_c"]]
    print(pvt_test_f1.to_string())

    print("\n" + "=" * 80)
    print("TABLE 3: FINAL HELD-OUT TEST ACCURACY (Evaluated once on 20% test split)")
    print("=" * 80)
    pvt_test_acc = df_summary.pivot(index="dataset_name", columns="method", values="test_accuracy")
    pvt_test_acc = pvt_test_acc[["random_search", "optuna_tpe", "metahpo_c"]]
    print(pvt_test_acc.to_string())

    print("\n" + "=" * 80)
    print("TABLE 4: EFFICIENCY PROGRESSION (Best-so-far CV Macro-F1 at trial checkpoints)")
    print("=" * 80)
    checkpoints = [1, 5, 10, 20]
    eff_records = []
    for ds in UNSEEN_DATASETS:
        for m in ["random_search", "optuna_tpe", "metahpo_c"]:
            sub = df_trials[(df_trials["dataset_name"] == ds) & (df_trials["method"] == m)].sort_values("trial_number")
            rec = {"dataset": ds, "method": m}
            for cp in checkpoints:
                # trial_number is 0-indexed, so trial checkpoint cp is at index cp-1
                val = sub.iloc[cp - 1]["best_so_far_cv_macro_f1"]
                rec[f"trial_{cp}"] = round(val, 6)
            eff_records.append(rec)
    df_eff = pd.DataFrame(eff_records)
    print(df_eff.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 5: RUNTIME COMPARISON (HPO Search vs Total Time in seconds)")
    print("=" * 80)
    pvt_time = df_summary[["dataset_name", "method", "hpo_runtime_seconds", "total_runtime_seconds"]]
    print(pvt_time.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 6: WARM-START CONFIGURATIONS RECOMMENDED BY METAHPO-C")
    print("=" * 80)
    for ds, cfgs in metadata["warm_start_configurations_evaluated"].items():
        print(f"\n--- Dataset: {ds} (Top {len(cfgs)} Warm-Start Configurations) ---")
        df_ws = pd.DataFrame(cfgs)
        print(df_ws.to_string(index=True))

    print("\n>>> ALL PHASE 10 CHECKS AND AUDITS PASSED SUCCESSFULLY. <<<\n")


if __name__ == "__main__":
    main()
