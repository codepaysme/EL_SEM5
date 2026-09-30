"""Verification script for Phase 2: OpenML WDBC loading and preprocessing."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, preprocess_features, split_dataset
from src.utils.logging_utils import get_logger

logger = get_logger("verify_wdbc_loader")


def main() -> None:
    """Execute WDBC dataset loading, validation, splitting, and preprocessing verification."""
    logger.info("=== Starting Phase 2 Verification: WDBC Dataset ===")

    # 1. Load WDBC dataset
    logger.info("Fetching WDBC dataset via OpenML registry lookup...")
    dataset = load_dataset("wdbc")

    # 2. Inspect and report dataset metadata
    print("\n" + "=" * 60)
    print("DATASET METADATA")
    print("=" * 60)
    print(f"Dataset Name               : {dataset.name}")
    print(f"Task Type                  : {dataset.task_type}")
    print(f"Number of Samples          : {dataset.n_samples}")
    print(f"Original Number of Features: {dataset.n_features}")
    print(f"Target Column Name         : {dataset.target_name}")

    class_counts = dataset.y.value_counts().to_dict()
    print(f"Number of Target Classes   : {len(class_counts)}")
    print(f"Class Distribution         : {class_counts}")
    print(f"Sample Feature Names       : {dataset.feature_names[:5]} ...")
    print("=" * 60)

    # 3. Create 80/20 Stratified Split
    logger.info("Creating 80/20 stratified train/test split (random_state=42)...")
    X_train, X_test, y_train, y_test = split_dataset(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=42,
    )

    print("\n" + "=" * 60)
    print("SPLIT DIMENSIONS")
    print("=" * 60)
    print(f"Train Set - X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"Test Set  - X shape: {X_test.shape}, y shape: {y_test.shape}")
    print(f"Train Class Distribution   : {y_train.value_counts().to_dict()}")
    print(f"Test Class Distribution    : {y_test.value_counts().to_dict()}")
    print("=" * 60)

    # 4. Fit Preprocessing Pipeline (strictly on train) & Transform
    logger.info("Applying leak-free preprocessing (imputation + encoding)...")
    X_train_proc, X_test_proc, preprocessor = preprocess_features(X_train, X_test)

    print("\n" + "=" * 60)
    print("PREPROCESSED FEATURE DIMENSIONS")
    print("=" * 60)
    print(f"Preprocessed X_train shape : {X_train_proc.shape}")
    print(f"Preprocessed X_test shape  : {X_test_proc.shape}")
    print(f"Fitted Preprocessor Steps  : {[name for name, _, _ in preprocessor.transformers_]}")
    print("=" * 60 + "\n")

    logger.info("=== Phase 2 Verification Completed Successfully! ===")


if __name__ == "__main__":
    main()
