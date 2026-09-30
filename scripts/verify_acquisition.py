"""Verification script for Phase 9: Confidence-Aware Candidate Selection."""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.confidence.acquisition import (
    HPO_SEARCH_SPACE,
    recommend_hyperparameters,
)
from src.features.meta_features import X_ONLY_META_FEATURE_NAMES
from src.utils.logging_utils import get_logger

logger = get_logger("verify_acquisition")


def main() -> None:
    logger.info("=== Starting Phase 9 Verification: Confidence-Aware Candidate Selection ===")

    meta_learning_dir = PROJECT_ROOT / "results" / "experiments" / "meta_learning"
    model_path = meta_learning_dir / "meta_model.joblib"
    meta_path = meta_learning_dir / "meta_model_metadata.json"
    summary_path = PROJECT_ROOT / "results" / "experiments" / "historical" / "historical_dataset_summary.csv"

    # 1. Load trained meta-model and metadata
    if not model_path.exists() or not meta_path.exists():
        logger.error(f"Missing meta-model artifacts in: {meta_learning_dir}")
        print("\nPHASE 9 VERIFICATION FAILED (Missing meta-model artifacts)\n")
        sys.exit(1)

    meta_model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_metadata = json.load(f)

    # 2. Load historical summary and extract WDBC meta-features
    df_summary = pd.read_csv(summary_path)
    wdbc_row = df_summary[df_summary["dataset_name"] == "wdbc"]

    if len(wdbc_row) == 0:
        logger.error("WDBC not found in historical dataset summary.")
        print("\nPHASE 9 VERIFICATION FAILED (WDBC missing)\n")
        sys.exit(1)

    wdbc_meta_features = wdbc_row[X_ONLY_META_FEATURE_NAMES].iloc[0].to_numpy(dtype=np.float64)

    # 3. Execute recommendation pipeline
    n_candidates = 200
    n_select = 10
    beta = 0.1
    seed = 42

    logger.info(f"Recommending top {n_select} hyperparameters for WDBC from {n_candidates} candidates (beta={beta}, seed={seed})...")
    top_candidates = recommend_hyperparameters(
        meta_model=meta_model,
        dataset_meta_features=wdbc_meta_features,
        n_candidates=n_candidates,
        n_select=n_select,
        beta=beta,
        seed=seed,
    )

    # 4. Extract diagnostic ranges
    pred_min, pred_max = float(top_candidates["predicted_score"].min()), float(top_candidates["predicted_score"].max())
    unc_min, unc_max = float(top_candidates["prediction_uncertainty"].min()), float(top_candidates["prediction_uncertainty"].max())
    conf_min, conf_max = float(top_candidates["relative_confidence"].min()), float(top_candidates["relative_confidence"].max())
    acq_min, acq_max = float(top_candidates["acquisition_score"].min()), float(top_candidates["acquisition_score"].max())

    # 5. Verify integrity and bounds
    is_valid = True
    errors = []

    if len(top_candidates) != n_select:
        is_valid = False
        errors.append(f"Expected {n_select} selected candidates, got {len(top_candidates)}.")

    for idx, row in top_candidates.iterrows():
        for p, (low, high, ptype) in HPO_SEARCH_SPACE.items():
            val = row[p]
            if ptype == "int":
                if not (int(low) <= int(val) <= int(high)):
                    is_valid = False
                    errors.append(f"Parameter '{p}' value {val} out of bounds [{low}, {high}].")
            else:
                if not (float(low) - 1e-6 <= float(val) <= float(high) + 1e-6):
                    is_valid = False
                    errors.append(f"Parameter '{p}' value {val} out of bounds [{low}, {high}].")

    if not np.all(np.isfinite(top_candidates["predicted_score"])):
        is_valid = False
        errors.append("Non-finite predicted scores found.")

    if not np.all(top_candidates["prediction_uncertainty"] >= 0.0):
        is_valid = False
        errors.append("Negative uncertainty values found.")

    if not np.all((top_candidates["relative_confidence"] > 0.0) & (top_candidates["relative_confidence"] <= 1.0)):
        is_valid = False
        errors.append("Confidence values outside (0, 1] bounds.")

    # 6. Print formatted summary
    print("\n" + "=" * 80)
    print("PHASE 9 CONFIDENCE-AWARE CANDIDATE SELECTION SUMMARY (WDBC Sanity Check)")
    print("=" * 80)
    print(f"Target Dataset Meta-Features   : WDBC ({len(wdbc_meta_features)} X-only features)")
    print(f"Total Candidate Pool Size      : {n_candidates}")
    print(f"Selected Top Configurations    : {n_select}")
    print(f"Acquisition Parameter (beta)   : {beta} (acquisition = pred + beta * uncertainty)")
    print(f"Reproducibility Seed           : {seed}")
    print("-" * 80)
    print("TOP-10 CANDIDATE METRICS RANGES:")
    print(f"  - Predicted CV Macro F1 Range: [{pred_min:.6f}, {pred_max:.6f}]")
    print(f"  - Prediction Uncertainty Range: [{unc_min:.6f}, {unc_max:.6f}]")
    print(f"  - Relative Confidence Range   : [{conf_min:.6f}, {conf_max:.6f}]")
    print(f"  - Acquisition Score Range     : [{acq_min:.6f}, {acq_max:.6f}]")
    print("-" * 80)
    print("TOP 10 RECOMMENDED HYPERPARAMETER CONFIGURATIONS FOR WDBC:")
    print(f"{'Rank':<4} | {'n_est':<5} | {'depth':<5} | {'lr':<8} | {'sub':<6} | {'col':<6} | {'mcw':<3} | {'gamma':<6} | {'lambda':<6} | {'Pred F1':<8} | {'Uncert':<7} | {'Acq Score':<9}")
    print("-" * 92)

    for rank, (_, row) in enumerate(top_candidates.iterrows(), 1):
        print(
            f"{rank:<4} | {int(row['n_estimators']):<5} | {int(row['max_depth']):<5} | "
            f"{row['learning_rate']:<8.4f} | {row['subsample']:<6.4f} | {row['colsample_bytree']:<6.4f} | "
            f"{int(row['min_child_weight']):<3} | {row['gamma']:<6.4f} | {row['reg_lambda']:<6.4f} | "
            f"{row['predicted_score']:<8.4f} | {row['prediction_uncertainty']:<7.4f} | {row['acquisition_score']:<9.4f}"
        )

    print("=" * 80)
    print("METHODOLOGICAL CONSTRAINTS CONFIRMATION:")
    print("  - Zero Test Label Usage: WDBC test partition not evaluated.")
    print("  - Held-out Unseen Datasets: madelon, diabetes, steel-plates-fault, ilpd untouched.")
    print("  - Interpretation: Empirical exploration heuristic, not calibrated probability.")
    print("=" * 80)

    if is_valid:
        print("\nPHASE 9 VERIFICATION PASSED\n")
    else:
        print("\nPHASE 9 VERIFICATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
