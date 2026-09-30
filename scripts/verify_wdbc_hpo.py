"""Verification script for Phase 4: Optuna + XGBoost HPO pipeline on WDBC."""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.loader import get_project_root, load_dataset, split_dataset
from src.hpo.optimizer import run_hpo
from src.utils.logging_utils import get_logger

logger = get_logger("verify_wdbc_hpo")


def main() -> None:
    """Execute WDBC 20-trial HPO experiment with Optuna and XGBoost."""
    logger.info("=== Starting Phase 4 Verification: Optuna + XGBoost HPO on WDBC ===")

    # 1. Load WDBC dataset
    logger.info("Loading WDBC dataset via data loader...")
    dataset = load_dataset("wdbc")

    # 2. Split dataset: 80% train (for CV inside Optuna) / 20% test (held out entirely)
    logger.info("Creating 80/20 stratified split (held-out test set kept strictly outside HPO)...")
    X_train, X_test, y_train, y_test = split_dataset(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=42,
    )

    print("\n" + "=" * 65)
    print("DATASET PARTITIONING (LEAK-FREE SETUP)")
    print("=" * 65)
    print(f"Dataset Name          : {dataset.name}")
    print(f"Total Samples         : {dataset.n_samples}")
    print(f"Train Portion (80%)   : {len(X_train)} samples (used for 3-fold CV in Optuna)")
    print(f"Held-out Test (20%)   : {len(X_test)} samples (UNTOUCHED by HPO)")
    print("=" * 65)

    # 3. Run HPO on training portion only
    output_dir = get_project_root() / "results" / "experiments"
    n_trials = 20
    random_state = 42

    logger.info(f"Running Optuna HPO ({n_trials} trials, Stratified 3-Fold CV, Macro F1)...")
    start_time = time.perf_counter()

    result = run_hpo(
        X=X_train,
        y=y_train,
        dataset_name=dataset.name,
        n_trials=n_trials,
        random_state=random_state,
        n_splits=3,
        output_dir=output_dir,
    )

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"HPO search completed in {elapsed_time:.2f} seconds.")

    # 4. Display Results
    print("\n" + "=" * 65)
    print("OPTUNA HPO EXPERIMENT RESULTS (WDBC)")
    print("=" * 65)
    print(f"Dataset               : {result.dataset_name}")
    print(f"Total Trials          : {result.n_trials}")
    print(f"Total Search Time     : {elapsed_time:.2f} seconds")
    print(f"Best Trial Index      : #{result.best_trial_number}")
    print(f"Best 3-Fold CV Macro F1: {result.best_score:.6f}")
    print("-" * 65)
    print("Optimal Hyperparameters:")
    for param_name, param_val in result.best_params.items():
        if isinstance(param_val, float):
            print(f"  - {param_name:<20}: {param_val:.6f}")
        else:
            print(f"  - {param_name:<20}: {param_val}")
    print("=" * 65)

    # 5. Confirm saved files
    csv_file = output_dir / f"{dataset.name}_hpo_trials.csv"
    json_file = output_dir / f"{dataset.name}_hpo_best.json"

    print("\n" + "=" * 65)
    print("SAVED EXPERIMENTAL ARTIFACTS")
    print("=" * 65)
    print(f"Trial Trajectory CSV  : {csv_file.relative_to(get_project_root())} (Exists: {csv_file.is_file()})")
    print(f"Best Result JSON      : {json_file.relative_to(get_project_root())} (Exists: {json_file.is_file()})")

    # Display Top 5 trials from the saved CSV
    if csv_file.is_file():
        df_saved = pd.read_csv(csv_file)
        top5 = df_saved.sort_values(by="cv_macro_f1", ascending=False).head(5)
        print("\nTop 5 Recorded Trials by CV Macro F1:")
        print(top5[["trial_number", "cv_macro_f1", "n_estimators", "max_depth", "learning_rate", "duration_seconds"]].to_string(index=False))

    print("=" * 65 + "\n")
    logger.info("=== Phase 4 Verification Completed Successfully! ===")


if __name__ == "__main__":
    main()
