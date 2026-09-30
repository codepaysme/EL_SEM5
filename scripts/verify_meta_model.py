"""Verification script for Phase 7: Meta-Learning Model Training and LODO Evaluation."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.meta_learning.meta_dataset import (
    META_INPUT_FEATURE_NAMES,
    META_TARGET_NAME,
    build_meta_dataset,
)
from src.meta_learning.meta_model import evaluate_lodo, train_final_meta_model
from src.utils.logging_utils import get_logger

logger = get_logger("verify_meta_model")


def main() -> None:
    logger.info("=== Starting Phase 7 Verification: Meta-Model Training & LODO Evaluation ===")

    output_dir = PROJECT_ROOT / "results" / "experiments" / "meta_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the Phase 6 meta-learning dataset
    meta_dataset = build_meta_dataset()
    meta_df = meta_dataset.full_df

    logger.info(f"Loaded meta-learning dataset: {len(meta_df)} examples across {meta_df['dataset_name'].nunique()} datasets.")

    # 2. Run Leave-One-Dataset-Out (LODO) evaluation
    logger.info("Running Leave-One-Dataset-Out (LODO) evaluation across 7 historical datasets...")
    lodo_result = evaluate_lodo(meta_df)
    lodo_pred_path, lodo_metrics_path = lodo_result.save(output_dir)

    # 3. Train final model on all 140 historical examples
    logger.info("Training final ExtraTrees meta-model on all 140 historical examples...")
    model_result = train_final_meta_model(meta_df, save_dir=output_dir)

    # 4. Print per-dataset metrics table
    print("\n" + "=" * 70)
    print("LEAVE-ONE-DATASET-OUT (LODO) REGRESSION PERFORMANCE")
    print("=" * 70)
    header = f"{'Dataset':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2':<10} | {'Spearman':<10}"
    print(header)
    print("-" * 70)

    for ds, metrics in lodo_result.per_dataset_metrics.items():
        print(f"{ds:<20} | {metrics['mae']:<10.6f} | {metrics['rmse']:<10.6f} | {metrics['r2']:<10.6f} | {metrics['spearman']:<10.6f}")

    print("=" * 70)
    print("\nOverall LODO:")
    print(f"MAE:      {lodo_result.overall_metrics['mae']:.6f}")
    print(f"RMSE:     {lodo_result.overall_metrics['rmse']:.6f}")
    print(f"R2:       {lodo_result.overall_metrics['r2']:.6f}")
    print(f"Spearman: {lodo_result.overall_metrics['spearman']:.6f}")

    print("\nBaseline (Training-Fold Target Mean):")
    print(f"MAE:      {lodo_result.baseline_overall_metrics['mae']:.6f}")
    print(f"RMSE:     {lodo_result.baseline_overall_metrics['rmse']:.6f}")
    print(f"R2:       {lodo_result.baseline_overall_metrics['r2']:.6f}")
    print(f"Spearman: {lodo_result.baseline_overall_metrics['spearman']:.6f}")
    print("=" * 70)

    # 5. Verify saved artifacts
    pred_df = pd.read_csv(lodo_pred_path)
    feat_imp_df = pd.read_csv(output_dir / "meta_feature_importance.csv")
    
    is_valid = True
    errors = []

    if len(pred_df) != 140:
        is_valid = False
        errors.append(f"lodo_predictions.csv has {len(pred_df)} rows, expected 140.")
    if not (output_dir / "lodo_metrics.json").exists():
        is_valid = False
        errors.append("lodo_metrics.json does not exist.")
    if not (output_dir / "meta_model.joblib").exists():
        is_valid = False
        errors.append("meta_model.joblib does not exist.")
    if not (output_dir / "meta_model_metadata.json").exists():
        is_valid = False
        errors.append("meta_model_metadata.json does not exist.")
    if len(feat_imp_df) != 30:
        is_valid = False
        errors.append(f"meta_feature_importance.csv has {len(feat_imp_df)} rows, expected 30.")

    print("\nARTIFACT AUDIT:")
    print(f"  - lodo_predictions.csv       : Exists ({len(pred_df)} rows)")
    print(f"  - lodo_metrics.json           : Exists")
    print(f"  - meta_model.joblib           : Exists")
    print(f"  - meta_model_metadata.json    : Exists")
    print(f"  - meta_feature_importance.csv : Exists ({len(feat_imp_df)} rows)")
    print("=" * 70)

    if is_valid:
        print("\nPHASE 7 VERIFICATION PASSED\n")
    else:
        print("\nPHASE 7 VERIFICATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
