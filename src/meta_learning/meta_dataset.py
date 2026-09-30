"""Meta-learning dataset construction and validation for MetaHPO-C.

Conceptual Architecture:
-------------------------
In MetaHPO-C, the meta-learning surrogate does NOT learn a direct mapping:
    Dataset -> Best Hyperparameters (m -> theta*)
Such direct regression suffers on small meta-training sample sizes and fails
to provide uncertainty estimates over the entire hyperparameter configuration space.

Instead, MetaHPO-C frames meta-learning as a supervised response-surface regression:
    (Dataset Meta-Features m, Candidate Hyperparameters theta) -> Expected CV Macro F1 y

Inputs (30 features):
- 22 dataset-level X-only meta-features (structural, statistical, correlation properties)
- 8 XGBoost candidate hyperparameters (n_estimators, max_depth, learning_rate, etc.)

Target (1 scalar):
- Observed Stratified 3-Fold Cross-Validation Macro F1 score

This formulation allows downstream components (Phase 7 & Phase 8) to:
1. Score arbitrary candidate hyperparameter configurations on unseen datasets.
2. Estimate prediction uncertainty / confidence across the search space.
3. Dynamically balance meta-informed warm-starting with Bayesian exploration.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.loader import HISTORICAL_DATASETS, get_project_root
from src.features.meta_features import X_ONLY_META_FEATURE_NAMES
from src.utils.logging_utils import get_logger

logger = get_logger("meta_dataset")

# 8 canonical XGBoost hyperparameter names explored in Phase 4 & Phase 5 HPO
HPO_PARAMETER_NAMES: List[str] = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_lambda",
]

# 30 predictive input features: 22 X-only meta-features + 8 hyperparameters
META_INPUT_FEATURE_NAMES: List[str] = X_ONLY_META_FEATURE_NAMES + HPO_PARAMETER_NAMES

# Optimization metric target
META_TARGET_NAME: str = "cv_macro_f1"

# Auxiliary traceability columns (excluded from predictive inputs)
AUXILIARY_COLUMNS: List[str] = [
    "dataset_name",
    "trial_number",
    "state",
    "duration_seconds",
]


def get_meta_input_feature_names() -> List[str]:
    """Get the canonical list of 30 predictive input feature names.

    Returns
    -------
    list of str
        22 X-only meta-features + 8 hyperparameter names.
    """
    return list(META_INPUT_FEATURE_NAMES)


def get_hpo_parameter_names() -> List[str]:
    """Get the canonical list of 8 XGBoost hyperparameter names.

    Returns
    -------
    list of str
        List of 8 hyperparameter column names.
    """
    return list(HPO_PARAMETER_NAMES)


def get_meta_target_name() -> str:
    """Get the target variable name for meta-learning regression.

    Returns
    -------
    str
        'cv_macro_f1'
    """
    return META_TARGET_NAME


@dataclass
class MetaDataset:
    """Container for the structured meta-learning experience dataset.

    Attributes
    ----------
    full_df : pd.DataFrame
        Complete DataFrame containing metadata, meta-features, hyperparameters, and target.
    X : pd.DataFrame
        Predictive feature matrix containing strictly the 30 input features.
    y : pd.Series
        Target vector containing the observed CV macro F1 scores.
    metadata : dict
        Metadata summary documenting construction details and dataset distributions.
    """

    full_df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    metadata: Dict[str, Any]

    def save(
        self,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Tuple[Path, Path, Path]:
        """Save the meta-learning dataset CSV, metadata JSON, and feature names JSON.

        Parameters
        ----------
        output_dir : Path or str, optional
            Output directory. Defaults to results/experiments/meta_learning.

        Returns
        -------
        tuple of (Path, Path, Path)
            Paths to (meta_dataset.csv, meta_dataset_metadata.json, meta_feature_names.json).
        """
        if output_dir is None:
            out_path = get_project_root() / "results" / "experiments" / "meta_learning"
        else:
            out_path = Path(output_dir).resolve()

        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Save full dataset CSV
        csv_path = out_path / "meta_dataset.csv"
        self.full_df.to_csv(csv_path, index=False)
        logger.info(f"Saved meta-learning dataset ({len(self.full_df)} rows) to: {csv_path.name}")

        # 2. Save metadata JSON
        json_path = out_path / "meta_dataset_metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved meta-dataset metadata to: {json_path.name}")

        # 3. Save feature names JSON
        feature_names_path = out_path / "meta_feature_names.json"
        feature_info = {
            "total_predictive_features": len(META_INPUT_FEATURE_NAMES),
            "meta_feature_count": len(X_ONLY_META_FEATURE_NAMES),
            "hyperparameter_count": len(HPO_PARAMETER_NAMES),
            "predictive_feature_names": META_INPUT_FEATURE_NAMES,
            "meta_feature_names": X_ONLY_META_FEATURE_NAMES,
            "hyperparameter_names": HPO_PARAMETER_NAMES,
            "target_name": META_TARGET_NAME,
            "auxiliary_columns": AUXILIARY_COLUMNS,
        }
        with open(feature_names_path, "w", encoding="utf-8") as f:
            json.dump(feature_info, f, indent=2)
        logger.info(f"Saved feature names ordering to: {feature_names_path.name}")

        return csv_path, json_path, feature_names_path


def validate_meta_dataset(
    df: pd.DataFrame,
    expected_datasets: Optional[List[str]] = None,
    expected_trials_per_dataset: int = 20,
) -> None:
    """Validate schema, completeness, integrity, and lack of leakage in a meta-learning DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    expected_datasets : list of str, optional
        List of expected historical dataset names. Defaults to HISTORICAL_DATASETS.
    expected_trials_per_dataset : int, default=20
        Expected number of trials per historical dataset.

    Raises
    ------
    ValueError
        If validation fails on shape, missing values, non-finite values, or leakage checks.
    """
    if expected_datasets is None:
        expected_datasets = list(HISTORICAL_DATASETS)

    expected_total_rows = len(expected_datasets) * expected_trials_per_dataset

    # 1. Total row count check
    if len(df) != expected_total_rows:
        raise ValueError(
            f"Meta-learning dataset row count mismatch: expected {expected_total_rows} rows "
            f"({len(expected_datasets)} datasets x {expected_trials_per_dataset} trials), found {len(df)}."
        )

    # 2. Dataset coverage and per-dataset counts
    if "dataset_name" not in df.columns:
        raise ValueError("Missing 'dataset_name' column in meta-learning dataset.")

    actual_datasets = sorted(df["dataset_name"].unique())
    sorted_expected = sorted(expected_datasets)
    if actual_datasets != sorted_expected:
        raise ValueError(
            f"Dataset mismatch: expected {sorted_expected}, found {actual_datasets}."
        )

    for ds in expected_datasets:
        ds_count = (df["dataset_name"] == ds).sum()
        if ds_count != expected_trials_per_dataset:
            raise ValueError(
                f"Dataset '{ds}' has {ds_count} examples, expected exactly {expected_trials_per_dataset}."
            )

    # 3. Check 22 X-only meta-features presence and finiteness
    for mf in X_ONLY_META_FEATURE_NAMES:
        if mf not in df.columns:
            raise ValueError(f"Missing canonical X-only meta-feature column: '{mf}'")
        if not np.issubdtype(df[mf].dtype, np.number):
            raise ValueError(f"Meta-feature column '{mf}' is not numeric (dtype: {df[mf].dtype}).")
        if df[mf].isnull().any():
            raise ValueError(f"Meta-feature column '{mf}' contains {df[mf].isnull().sum()} null values.")
        if not np.all(np.isfinite(df[mf])):
            raise ValueError(f"Meta-feature column '{mf}' contains non-finite values (inf or nan).")

    # 4. Check 8 hyperparameters presence and finiteness
    for hp in HPO_PARAMETER_NAMES:
        if hp not in df.columns:
            raise ValueError(f"Missing hyperparameter column: '{hp}'")
        if not np.issubdtype(df[hp].dtype, np.number):
            raise ValueError(f"Hyperparameter column '{hp}' is not numeric (dtype: {df[hp].dtype}).")
        if df[hp].isnull().any():
            raise ValueError(f"Hyperparameter column '{hp}' contains {df[hp].isnull().sum()} null values.")
        if not np.all(np.isfinite(df[hp])):
            raise ValueError(f"Hyperparameter column '{hp}' contains non-finite values (inf or nan).")

    # 5. Check target column presence and finiteness
    if META_TARGET_NAME not in df.columns:
        raise ValueError(f"Missing meta-learning target column: '{META_TARGET_NAME}'")
    if not np.issubdtype(df[META_TARGET_NAME].dtype, np.number):
        raise ValueError(f"Target column '{META_TARGET_NAME}' is not numeric.")
    if df[META_TARGET_NAME].isnull().any():
        raise ValueError(f"Target column '{META_TARGET_NAME}' contains null values.")
    if not np.all(np.isfinite(df[META_TARGET_NAME])):
        raise ValueError(f"Target column '{META_TARGET_NAME}' contains non-finite values.")

    # 6. Leakage and predictive feature boundary protection
    # Ensure no target-dependent meta-features are present in META_INPUT_FEATURE_NAMES
    target_dependent_forbidden = [
        "n_classes",
        "majority_class_ratio",
        "minority_class_ratio",
        "class_entropy",
        "n_classes_descriptive",
        "n_trials",
    ]
    for forbidden in target_dependent_forbidden:
        if forbidden in META_INPUT_FEATURE_NAMES:
            raise ValueError(f"Leakage violation: forbidden feature '{forbidden}' found in META_INPUT_FEATURE_NAMES.")

    # Ensure no auxiliary/metadata columns are in META_INPUT_FEATURE_NAMES
    for aux in AUXILIARY_COLUMNS + [META_TARGET_NAME]:
        if aux in META_INPUT_FEATURE_NAMES:
            raise ValueError(f"Leakage violation: metadata column '{aux}' found in META_INPUT_FEATURE_NAMES.")

    # 7. Check total predictive feature count
    if len(META_INPUT_FEATURE_NAMES) != 30:
        raise ValueError(
            f"Expected exactly 30 predictive features (22 meta + 8 hparam), got {len(META_INPUT_FEATURE_NAMES)}."
        )


def build_meta_dataset(
    trials_path: Optional[Union[Path, str]] = None,
    summary_path: Optional[Union[Path, str]] = None,
    save_dir: Optional[Union[Path, str]] = None,
) -> MetaDataset:
    """Construct a clean, leak-free supervised meta-learning dataset from historical HPO artifacts.

    Parameters
    ----------
    trials_path : Path or str, optional
        Path to historical_hpo_trials.csv. Defaults to results/experiments/historical/historical_hpo_trials.csv.
    summary_path : Path or str, optional
        Path to historical_dataset_summary.csv. Defaults to results/experiments/historical/historical_dataset_summary.csv.
    save_dir : Path or str, optional
        If provided, saves meta_dataset.csv, meta_dataset_metadata.json, and meta_feature_names.json to this directory.

    Returns
    -------
    MetaDataset
        Constructed and validated MetaDataset container.
    """
    root = get_project_root()
    if trials_path is None:
        trials_path = root / "results" / "experiments" / "historical" / "historical_hpo_trials.csv"
    else:
        trials_path = Path(trials_path).resolve()

    if summary_path is None:
        summary_path = root / "results" / "experiments" / "historical" / "historical_dataset_summary.csv"
    else:
        summary_path = Path(summary_path).resolve()

    if not trials_path.exists():
        raise FileNotFoundError(f"Historical trials CSV not found at: {trials_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Historical dataset summary CSV not found at: {summary_path}")

    logger.info(f"Loading historical trials from: {trials_path.name}")
    df_trials = pd.read_csv(trials_path)

    logger.info(f"Loading historical dataset summary from: {summary_path.name}")
    df_summary = pd.read_csv(summary_path)

    # 1. Extract strictly the canonical 22 X-only meta-features from the summary
    summary_cols = ["dataset_name"] + list(X_ONLY_META_FEATURE_NAMES)
    missing_summary_cols = [col for col in summary_cols if col not in df_summary.columns]
    if missing_summary_cols:
        raise ValueError(f"Summary CSV missing required columns: {missing_summary_cols}")

    df_meta_clean = df_summary[summary_cols].drop_duplicates(subset=["dataset_name"])

    # 2. Join meta-features onto historical HPO trials by dataset_name
    df_merged = pd.merge(
        df_trials,
        df_meta_clean,
        on="dataset_name",
        how="inner",
    )

    if len(df_merged) != len(df_trials):
        raise ValueError(
            f"Join integrity error: merged {len(df_merged)} rows from {len(df_trials)} trial records."
        )

    # 3. Cleanly structure column ordering: Metadata -> 22 Meta-features -> 8 Hyperparameters -> Target
    present_aux = [c for c in AUXILIARY_COLUMNS if c in df_merged.columns]
    ordered_columns = present_aux + X_ONLY_META_FEATURE_NAMES + HPO_PARAMETER_NAMES + [META_TARGET_NAME]
    df_meta_dataset = df_merged[ordered_columns].copy()

    # 4. Validate dataset integrity and leak-free boundary
    validate_meta_dataset(df_meta_dataset)

    # 5. Extract predictive feature matrix X and target y
    X = df_meta_dataset[META_INPUT_FEATURE_NAMES].copy()
    y = df_meta_dataset[META_TARGET_NAME].copy()

    # 6. Build comprehensive metadata description
    datasets_list = sorted(df_meta_dataset["dataset_name"].unique())
    examples_per_dataset = df_meta_dataset.groupby("dataset_name").size().to_dict()

    metadata = {
        "description": "MetaHPO-C Supervised Meta-Learning Dataset (Meta-features + Hyperparameters -> Validation F1)",
        "total_examples": int(len(df_meta_dataset)),
        "total_predictive_features": int(len(META_INPUT_FEATURE_NAMES)),
        "meta_feature_count": int(len(X_ONLY_META_FEATURE_NAMES)),
        "hyperparameter_count": int(len(HPO_PARAMETER_NAMES)),
        "target_column": META_TARGET_NAME,
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "historical_datasets": datasets_list,
        "examples_per_dataset": examples_per_dataset,
        "predictive_feature_names": META_INPUT_FEATURE_NAMES,
        "meta_feature_names": X_ONLY_META_FEATURE_NAMES,
        "hyperparameter_names": HPO_PARAMETER_NAMES,
        "excluded_auxiliary_columns": present_aux,
    }

    meta_dataset = MetaDataset(
        full_df=df_meta_dataset,
        X=X,
        y=y,
        metadata=metadata,
    )

    # 7. Optionally save artifacts if save_dir is specified
    if save_dir is not None:
        meta_dataset.save(save_dir)

    return meta_dataset
