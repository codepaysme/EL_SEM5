"""Verification script for Phase 6: Supervised Meta-Learning Dataset Construction."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.meta_learning.meta_dataset import (
    AUXILIARY_COLUMNS,
    HPO_PARAMETER_NAMES,
    META_INPUT_FEATURE_NAMES,
    META_TARGET_NAME,
    build_meta_dataset,
)
from src.utils.logging_utils import get_logger

logger = get_logger("verify_meta_dataset")


def main() -> None:
    logger.info("=== Starting Phase 6 Verification: Meta-Learning Dataset Construction ===")

    output_dir = PROJECT_ROOT / "results" / "experiments" / "meta_learning"
    
    # 1. Build and persist the meta-learning dataset
    meta_dataset = build_meta_dataset(save_dir=output_dir)

    df = meta_dataset.full_df
    X = meta_dataset.X
    y = meta_dataset.y
    metadata = meta_dataset.metadata

    # 2. Extract verification statistics
    n_examples = len(df)
    n_input_features = X.shape[1]
    n_meta_features = len(metadata["meta_feature_names"])
    n_hyperparameters = len(metadata["hyperparameter_names"])
    target_name = metadata["target_column"]
    target_min = float(y.min())
    target_max = float(y.max())
    target_mean = float(y.mean())
    target_std = float(y.std())

    # Finiteness & missingness checks
    missing_input_count = int(X.isnull().sum().sum())
    non_finite_input_count = int((~np.isfinite(X.to_numpy())).sum())
    missing_target_count = int(y.isnull().sum())
    non_finite_target_count = int((~np.isfinite(y.to_numpy())).sum())

    # Dataset distribution
    ds_counts = df.groupby("dataset_name").size().to_dict()

    print("\n" + "=" * 75)
    print("META-LEARNING DATASET SPECIFICATION & VERIFICATION SUMMARY")
    print("=" * 75)
    print(f"Total Supervised Examples : {n_examples} (7 datasets x 20 trials)")
    print(f"Total Predictive Inputs   : {n_input_features} (22 Meta-features + 8 Hyperparameters)")
    print(f"  - Dataset Meta-Features : {n_meta_features} (X-only, zero target leakage)")
    print(f"  - Hyperparameters       : {n_hyperparameters} (XGBoost configuration space)")
    print(f"Supervised Target Metric  : '{target_name}' (Stratified 3-Fold CV Macro F1)")
    print(f"  - Min / Max Score       : {target_min:.6f} / {target_max:.6f}")
    print(f"  - Mean +/- Std Score     : {target_mean:.6f} +/- {target_std:.6f}")
    print("-" * 75)
    print("DATASET LEVEL EXAMPLE BREAKDOWN:")
    for ds_name, count in ds_counts.items():
        ds_y = df[df["dataset_name"] == ds_name][target_name]
        print(f"  * {ds_name:18s}: {count:2d} trials | CV Macro F1: [{ds_y.min():.4f}, {ds_y.max():.4f}]")
    print("-" * 75)
    print("DATA QUALITY & INTEGRITY AUDIT:")
    print(f"  - Missing input values  : {missing_input_count}")
    print(f"  - Non-finite inputs     : {non_finite_input_count}")
    print(f"  - Missing target values : {missing_target_count}")
    print(f"  - Non-finite targets    : {non_finite_target_count}")
    print(f"  - Auxiliary columns     : {metadata['excluded_auxiliary_columns']}")
    print("-" * 75)
    print(f"SAVED ARTIFACTS IN: {output_dir}")
    print(f"  - CSV Dataset : meta_dataset.csv ({n_examples} rows, {df.shape[1]} columns)")
    print(f"  - Metadata    : meta_dataset_metadata.json")
    print(f"  - Feature Info: meta_feature_names.json")
    print("=" * 75)

    is_valid = (
        n_examples == 140
        and n_input_features == 30
        and n_meta_features == 22
        and n_hyperparameters == 8
        and missing_input_count == 0
        and non_finite_input_count == 0
        and missing_target_count == 0
        and non_finite_target_count == 0
        and len(ds_counts) == 7
        and all(c == 20 for c in ds_counts.values())
    )

    if is_valid:
        print("\nFINAL RESULT: META-LEARNING DATASET VERIFIED SUCCESSFULLY [PASS]\n")
    else:
        print("\nFINAL RESULT: META-LEARNING DATASET VERIFICATION FAILED [FAIL]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
