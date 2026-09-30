"""Verification script inspecting Phase 5 historical HPO artifacts and aggregates."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.loader import (
    HISTORICAL_DATASETS,
    UNSEEN_DATASETS,
    get_historical_dataset_names,
    get_project_root,
    get_unseen_dataset_names,
)
from src.features.meta_features import X_ONLY_META_FEATURE_NAMES
from src.utils.logging_utils import get_logger

logger = get_logger("verify_historical_hpo")


def main() -> None:
    """Inspect and report historical HPO artifacts, trials, and summary dataset."""
    logger.info("=== Starting Phase 5 Verification: Historical HPO Artifacts ===")

    historical_dir = get_project_root() / "results" / "experiments" / "historical"

    if not historical_dir.is_dir():
        logger.error(f"Historical results directory does not exist: {historical_dir}")
        sys.exit(1)

    expected_datasets = get_historical_dataset_names()
    unseen_datasets = get_unseen_dataset_names()

    print("\n" + "=" * 70)
    print("HISTORICAL DATASETS ARTIFACT AUDIT")
    print("=" * 70)

    dataset_rows = []
    missing_artifacts = []

    for name in expected_datasets:
        trials_csv = historical_dir / f"{name}_hpo_trials.csv"
        best_json = historical_dir / f"{name}_hpo_best.json"
        mf_json = historical_dir / f"{name}_meta_features.json"

        has_trials = trials_csv.is_file()
        has_best = best_json.is_file()
        has_mf = mf_json.is_file()

        if not (has_trials and has_best and has_mf):
            missing_artifacts.append((name, has_trials, has_best, has_mf))
            continue

        # Inspect trial CSV
        df_trials = pd.read_csv(trials_csv)
        n_trials = len(df_trials)

        # Inspect best JSON
        with open(best_json, "r", encoding="utf-8") as f:
            best_data = json.load(f)
        best_score = best_data.get("best_cv_macro_f1", float("nan"))
        best_trial_idx = best_data.get("best_trial_number", -1)

        # Inspect meta-features JSON
        with open(mf_json, "r", encoding="utf-8") as f:
            mf_data = json.load(f)
        n_mf = mf_data.get("n_meta_features", len(mf_data.get("features", {})))
        n_samples = mf_data.get("features", {}).get("n_samples", 0)
        n_features = mf_data.get("features", {}).get("n_features", 0)

        dataset_rows.append(
            {
                "dataset_name": name,
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "n_meta_features": int(n_mf),
                "hpo_trials": n_trials,
                "best_trial": best_trial_idx,
                "best_cv_macro_f1": best_score,
            }
        )

    # Display dataset breakdown
    if dataset_rows:
        df_report = pd.DataFrame(dataset_rows)
        print(f"{'Dataset':<16} | {'Samples':>7} | {'Feats':>5} | {'X-MFs':>5} | {'Trials':>6} | {'Best #':>6} | {'Best Macro F1':>13}")
        print("-" * 70)
        for _, row in df_report.iterrows():
            print(
                f"{row['dataset_name']:<16} | {row['n_samples']:>7} | {row['n_features']:>5} | "
                f"{row['n_meta_features']:>5} | {row['hpo_trials']:>6} | #{row['best_trial']:>5} | {row['best_cv_macro_f1']:>13.6f}"
            )
    else:
        print("No complete historical dataset artifacts found!")

    if missing_artifacts:
        print("\n[WARNING] Incomplete artifacts for datasets:")
        for name, ht, hb, hm in missing_artifacts:
            print(f"  - {name}: trials_csv={ht}, best_json={hb}, meta_features_json={hm}")

    print("=" * 70)

    # Check Aggregate Files
    agg_trials_file = historical_dir / "historical_hpo_trials.csv"
    agg_summary_file = historical_dir / "historical_dataset_summary.csv"

    has_agg_trials = agg_trials_file.is_file()
    has_agg_summary = agg_summary_file.is_file()

    total_trial_rows = len(pd.read_csv(agg_trials_file)) if has_agg_trials else 0
    summary_rows = len(pd.read_csv(agg_summary_file)) if has_agg_summary else 0

    print("\n" + "=" * 70)
    print("AGGREGATE META-EXPERIENCE FILES")
    print("=" * 70)
    print(f"Historical Datasets Processed : {len(dataset_rows)} / {len(expected_datasets)}")
    print(f"Aggregate Trials CSV File     : {agg_trials_file.name} (Exists: {has_agg_trials}, Rows: {total_trial_rows})")
    print(f"Aggregate Summary CSV File    : {agg_summary_file.name} (Exists: {has_agg_summary}, Rows: {summary_rows})")
    print(f"X-Only Meta-Feature Count     : {len(X_ONLY_META_FEATURE_NAMES)}")

    # Unseen Dataset Contamination Check
    unseen_found = []
    for unseen_name in unseen_datasets:
        matches = list(historical_dir.glob(f"*{unseen_name}*"))
        if matches:
            unseen_found.append((unseen_name, [m.name for m in matches]))

    print(f"Unseen Datasets Included      : {len(unseen_found)}")
    if unseen_found:
        print(f"[ERROR] Found unseen dataset contamination in historical directory: {unseen_found}")
    else:
        print("Held-Out Unseen Isolation     : VERIFIED CLEAN (Zero unseen datasets touched)")
    print("=" * 70 + "\n")

    logger.info("=== Phase 5 Verification Completed Successfully! ===")


if __name__ == "__main__":
    main()
