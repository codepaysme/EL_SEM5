"""Historical HPO experiment runner across the 7 meta-training datasets."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.data.loader import (
    HISTORICAL_DATASETS,
    UNSEEN_DATASETS,
    get_historical_dataset_names,
    get_project_root,
    get_unseen_dataset_names,
    load_dataset,
    split_dataset,
)
from src.features.meta_features import (
    X_ONLY_META_FEATURE_NAMES,
    extract_meta_features,
    get_x_only_meta_feature_names,
)
from src.hpo.optimizer import HPOResult, run_hpo
from src.utils.logging_utils import get_logger

logger = get_logger("run_historical_hpo")


def save_meta_features_json(
    meta_features_dict: Dict[str, Any],
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Save 22-dimensional X-only meta-features as structured JSON.

    Parameters
    ----------
    meta_features_dict : dict
        Dictionary of X-only meta-feature values.
    dataset_name : str
        Name of the dataset.
    output_dir : Path
        Target directory.

    Returns
    -------
    Path
        Path to the saved JSON artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{dataset_name}_meta_features.json"

    # Filter strictly to the 22 canonical X-only features in canonical order
    canonical_features = {
        name: float(meta_features_dict[name])
        for name in X_ONLY_META_FEATURE_NAMES
        if meta_features_dict.get(name) is not None
    }

    payload = {
        "dataset_name": dataset_name,
        "n_meta_features": len(canonical_features),
        "feature_names": get_x_only_meta_feature_names(),
        "features": canonical_features,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path


def run_historical_dataset(
    dataset_name: str,
    n_trials: int = 20,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute meta-feature extraction, splitting, and HPO for a single historical dataset.

    Parameters
    ----------
    dataset_name : str
        Historical dataset identifier.
    n_trials : int, default=20
        Number of Optuna trials.
    random_state : int, default=42
        Reproducibility seed.
    output_dir : Path, optional
        Destination directory for dataset artifacts.

    Returns
    -------
    dict
        Summary record containing dataset metadata, 22 X-only meta-features, and HPO result.
    """
    if output_dir is None:
        output_dir = get_project_root() / "results" / "experiments" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unseen dataset guard
    if dataset_name in UNSEEN_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' is reserved as a held-out unseen test dataset. "
            "It must NOT be processed in the historical training pool."
        )

    logger.info(f"[{dataset_name}] Loading dataset from registry...")
    dataset = load_dataset(dataset_name)

    # 2. Extract 22 X-only meta-features (strictly zero label leakage)
    logger.info(f"[{dataset_name}] Extracting 22 X-only meta-features (y=None)...")
    mf = extract_meta_features(dataset.X, y=None, dataset_name=dataset_name)
    mf_json_path = save_meta_features_json(mf.values, dataset_name, output_dir)
    logger.info(f"[{dataset_name}] Saved meta-features to: {mf_json_path.name}")

    # 3. Create 80/20 stratified split (20% held-out test data is untouched by HPO)
    logger.info(f"[{dataset_name}] Partitioning: 80% train ({int(0.8*dataset.n_samples)} samples), 20% held-out test ({int(0.2*dataset.n_samples)} samples)...")
    X_train, X_test, y_train, y_test = split_dataset(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=random_state,
    )

    # 4. Run Optuna HPO on 80% training slice with Stratified 3-Fold CV
    logger.info(f"[{dataset_name}] Starting Optuna HPO ({n_trials} trials, 3-Fold CV, Macro F1)...")
    t0 = time.perf_counter()
    hpo_result = run_hpo(
        X=X_train,
        y=y_train,
        dataset_name=dataset_name,
        n_trials=n_trials,
        random_state=random_state,
        n_splits=3,
        output_dir=output_dir,
    )
    duration = time.perf_counter() - t0
    logger.info(
        f"[{dataset_name}] Completed {n_trials} trials in {duration:.2f}s | "
        f"Best Trial #{hpo_result.best_trial_number} CV Macro F1: {hpo_result.best_score:.6f}"
    )

    # Descriptive class count for reporting
    n_classes_descriptive = len(dataset.y.dropna().unique())

    # Build summary record combining metadata, HPO outcome, and all 22 X-only features
    summary_record: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "n_classes_descriptive": n_classes_descriptive,
        "best_cv_macro_f1": float(hpo_result.best_score),
        "best_trial_number": int(hpo_result.best_trial_number),
        "n_trials": int(hpo_result.n_trials),
    }

    # Add all 22 canonical X-only meta-features
    for feat_name in X_ONLY_META_FEATURE_NAMES:
        summary_record[feat_name] = float(mf.values[feat_name])

    return {
        "dataset_name": dataset_name,
        "hpo_result": hpo_result,
        "summary_record": summary_record,
        "duration_seconds": duration,
    }


def run_all_historical_datasets(
    dataset_names: Optional[List[str]] = None,
    n_trials: int = 20,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> None:
    """Run HPO and meta-feature collection across all specified historical datasets and build aggregates.

    Parameters
    ----------
    dataset_names : list of str, optional
        List of historical dataset names. Defaults to the 7 canonical historical datasets.
    n_trials : int, default=20
        Number of trials per dataset.
    random_state : int, default=42
        Reproducibility seed.
    output_dir : Path, optional
        Destination directory.
    """
    if output_dir is None:
        output_dir = get_project_root() / "results" / "experiments" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_names is None:
        dataset_names = get_historical_dataset_names()

    # Pre-validation: verify no unseen datasets are requested
    for name in dataset_names:
        if name in UNSEEN_DATASETS:
            raise ValueError(
                f"Dataset '{name}' is in the reserved unseen test pool. "
                "Historical HPO runners must NOT touch unseen test datasets."
            )

    logger.info(f"=== Starting Historical HPO Execution: {len(dataset_names)} Datasets ===")
    logger.info(f"Target datasets: {dataset_names}")
    logger.info(f"Trials per dataset: {n_trials} | Random state: {random_state}")

    all_trials_dfs: List[pd.DataFrame] = []
    all_summary_records: List[Dict[str, Any]] = []

    total_start_time = time.perf_counter()

    for idx, name in enumerate(dataset_names, 1):
        logger.info(f"\n--- Processing Dataset [{idx}/{len(dataset_names)}]: {name} ---")
        try:
            res = run_historical_dataset(
                dataset_name=name,
                n_trials=n_trials,
                random_state=random_state,
                output_dir=output_dir,
            )
            all_trials_dfs.append(res["hpo_result"].to_dataframe())
            all_summary_records.append(res["summary_record"])
        except Exception as exc:
            logger.error(f"Failed processing historical dataset '{name}': {exc}", exc_info=True)
            raise RuntimeError(f"Historical HPO aborted due to failure on '{name}': {exc}") from exc

    total_duration = time.perf_counter() - total_start_time

    # 1. Save Aggregated Trial Trajectory CSV
    if all_trials_dfs:
        df_combined_trials = pd.concat(all_trials_dfs, ignore_index=True)
        aggregate_trials_csv = output_dir / "historical_hpo_trials.csv"
        df_combined_trials.to_csv(aggregate_trials_csv, index=False)
        logger.info(f"Saved combined trials ({len(df_combined_trials)} rows) to: {aggregate_trials_csv.name}")

    # 2. Save Aggregated Historical Dataset Summary CSV
    if all_summary_records:
        df_summary = pd.DataFrame(all_summary_records)
        aggregate_summary_csv = output_dir / "historical_dataset_summary.csv"
        df_summary.to_csv(aggregate_summary_csv, index=False)
        logger.info(f"Saved historical dataset summary ({len(df_summary)} datasets) to: {aggregate_summary_csv.name}")

    logger.info(f"\n=== Historical HPO Pipeline Completed in {total_duration:.2f}s ===")


def main() -> None:
    """CLI entry point for running historical HPO experiments."""
    parser = argparse.ArgumentParser(description="Run HPO and meta-feature collection for historical datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of historical dataset names to process (default: all 7 historical datasets).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna HPO trials per dataset (default: 20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for historical artifacts (default: results/experiments/historical).",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve() if args.output_dir else None

    run_all_historical_datasets(
        dataset_names=args.datasets,
        n_trials=args.n_trials,
        random_state=args.random_state,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
