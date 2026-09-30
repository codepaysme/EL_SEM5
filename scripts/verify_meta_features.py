"""Verification script for Phase 3: Meta-feature extraction on WDBC (X-only and Labeled)."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.data.loader import load_dataset
from src.features.meta_features import (
    TARGET_META_FEATURE_NAMES,
    X_ONLY_META_FEATURE_NAMES,
    extract_meta_features,
    meta_features_to_array,
)
from src.utils.logging_utils import get_logger

logger = get_logger("verify_meta_features")


def main() -> None:
    """Execute WDBC meta-feature extraction for both X-only and labeled modes."""
    logger.info("=== Starting Phase 3 Verification: Meta-Feature Extraction ===")

    # 1. Load WDBC dataset
    logger.info("Loading WDBC dataset via data loader...")
    dataset = load_dataset("wdbc")

    # --------------------------------------------------------------------------
    # 2. Extract X-Only Meta-Features (Unseen Dataset Recommendation Mode)
    # --------------------------------------------------------------------------
    logger.info("Extracting X-only meta-features (y=None, zero label leakage)...")
    mf_unlabeled = extract_meta_features(dataset.X, y=None, dataset_name=dataset.name)
    arr_x_only = meta_features_to_array(mf_unlabeled, include_target_features=False)

    print("\n" + "=" * 65)
    print("1. UNLABELED / X-ONLY META-FEATURES (Used for Unseen Datasets)")
    print("=" * 65)
    print(f"{'Meta-Feature':<35} | {'Value':>25}")
    print("-" * 65)
    for name in X_ONLY_META_FEATURE_NAMES:
        val = mf_unlabeled.values[name]
        formatted_val = f"{val:>25.6f}" if isinstance(val, float) else f"{str(val):>25}"
        print(f"{name:<35} | {formatted_val}")
    print("-" * 65)
    print(f"Number of X-Only Meta-Features : {len(X_ONLY_META_FEATURE_NAMES)}")
    print(f"All X-Only Values Finite       : {all(np.isfinite(arr_x_only))}")
    print(f"NumPy Vector Shape             : {arr_x_only.shape}")
    print(f"NumPy Vector Dtype             : {arr_x_only.dtype}")
    print("=" * 65)

    # --------------------------------------------------------------------------
    # 3. Extract Full Labeled Meta-Features (Historical Datasets Mode)
    # --------------------------------------------------------------------------
    logger.info("Extracting full labeled meta-features (with ground-truth labels)...")
    mf_labeled = extract_meta_features(dataset.X, y=dataset.y, dataset_name=dataset.name)
    arr_labeled = meta_features_to_array(mf_labeled, include_target_features=True)

    print("\n" + "=" * 65)
    print("2. TARGET-DEPENDENT DESCRIPTIVE FEATURES (Historical Datasets)")
    print("=" * 65)
    print(f"{'Target Meta-Feature':<35} | {'Value':>25}")
    print("-" * 65)
    for name in TARGET_META_FEATURE_NAMES:
        val = mf_labeled.values[name]
        formatted_val = f"{val:>25.6f}" if isinstance(val, float) else f"{str(val):>25}"
        print(f"{name:<35} | {formatted_val}")
    print("-" * 65)
    print(f"Number of Target Meta-Features : {len(TARGET_META_FEATURE_NAMES)}")
    print(f"Total Combined Features        : {len(arr_labeled)}")
    print(f"NumPy Labeled Vector Shape     : {arr_labeled.shape}")
    print(f"All Labeled Values Finite      : {all(np.isfinite(arr_labeled))}")
    print("=" * 65 + "\n")

    logger.info("=== Phase 3 Verification Completed Successfully! ===")


if __name__ == "__main__":
    main()
