"""Verification script for Phase 8: Prediction Uncertainty & Relative Confidence Estimation."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.confidence.uncertainty import evaluate_lodo_uncertainty
from src.meta_learning.meta_dataset import build_meta_dataset
from src.utils.logging_utils import get_logger

logger = get_logger("verify_uncertainty")


def main() -> None:
    logger.info("=== Starting Phase 8 Verification: Uncertainty & Confidence Estimation ===")

    output_dir = PROJECT_ROOT / "results" / "experiments" / "meta_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the Phase 6 meta-learning dataset
    meta_dataset = build_meta_dataset()
    meta_df = meta_dataset.full_df

    # 2. Run LODO uncertainty evaluation
    logger.info("Running LODO uncertainty evaluation across 7 historical datasets...")
    result = evaluate_lodo_uncertainty(meta_df)
    csv_path, json_path = result.save(output_dir)

    df_preds = result.predictions_df
    m = result.metrics

    # 3. Sanity and integrity checks
    is_valid = True
    errors = []

    if len(df_preds) != 140:
        is_valid = False
        errors.append(f"Expected 140 prediction rows, found {len(df_preds)}.")

    unc = df_preds["prediction_uncertainty"].to_numpy()
    conf = df_preds["relative_confidence"].to_numpy()
    err = df_preds["absolute_error"].to_numpy()

    if not np.all(np.isfinite(unc)):
        is_valid = False
        errors.append("Non-finite values found in prediction_uncertainty.")

    if not np.all(unc >= 0):
        is_valid = False
        errors.append("Negative values found in prediction_uncertainty.")

    if not np.all((conf > 0) & (conf <= 1.0)):
        is_valid = False
        errors.append("Confidence values outside (0, 1] range.")

    if not np.all(np.isfinite(err)):
        is_valid = False
        errors.append("Non-finite values found in absolute_error.")

    # 4. Print structured summary report
    print("\n" + "=" * 70)
    print("PHASE 8 UNCERTAINTY & CONFIDENCE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"LODO examples evaluated        : {len(df_preds)} (7 datasets x 20 trials)")
    print(f"Ensemble estimators (T)        : {m['n_estimators']}")
    print(f"Mean uncertainty (std proxy)   : {m['overall_mean_uncertainty']:.6f}")
    print(f"Median uncertainty (std proxy) : {m['overall_median_uncertainty']:.6f}")
    print(f"Mean absolute error (MAE)      : {m['overall_mean_absolute_error']:.6f}")
    print(f"Median absolute error          : {m['overall_median_absolute_error']:.6f}")
    print("-" * 70)
    print("UNCERTAINTY VS. ERROR CORRELATION:")
    print(f"  - Spearman(uncertainty, error) : {m['overall_spearman_uncertainty_vs_absolute_error']:+.6f}")
    print(f"  - Pearson(uncertainty, error)  : {m['overall_pearson_uncertainty_vs_absolute_error']:+.6f}")
    print("-" * 70)
    print("UNCERTAINTY RANKING DIAGNOSTIC (ERROR BY UNCERTAINTY HALF):")
    print(f"  - Low-uncertainty half MAE   : {m['low_uncertainty_mae']:.6f}")
    print(f"  - High-uncertainty half MAE  : {m['high_uncertainty_mae']:.6f}")
    print("-" * 70)
    print("PER-DATASET UNCERTAINTY BREAKDOWN:")
    print(f"{'Dataset':<18} | {'Mean Unc':<10} | {'Median Unc':<10} | {'Low MAE':<10} | {'High MAE':<10} | {'Spearman':<10}")
    print("-" * 75)
    for ds, dm in result.per_dataset_metrics.items():
        print(f"{ds:<18} | {dm['mean_uncertainty']:<10.6f} | {dm['median_uncertainty']:<10.6f} | {dm['low_uncertainty_mae']:<10.6f} | {dm['high_uncertainty_mae']:<10.6f} | {dm['spearman_uncertainty_vs_error']:<+10.4f}")
    print("-" * 75)
    print("METHODOLOGICAL CLASSIFICATION:")
    print("  - Calibration : NOT CLAIMED (Relative heuristic for configuration ranking)")
    print("  - Method      : ExtraTrees ensemble prediction dispersion (ddof=0)")
    print("=" * 70)

    if is_valid:
        print("\nPHASE 8 VERIFICATION PASSED\n")
    else:
        print("\nPHASE 8 VERIFICATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
