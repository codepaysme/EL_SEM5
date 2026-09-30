"""Dataset-level meta-feature extraction for classification datasets.

Supports both:
1. X-only meta-features (22 features): computed purely from input features X,
   guaranteeing zero target leakage for unseen dataset recommendations.
2. Target-dependent descriptive features (4 features): class balance and entropy
   computed only when ground-truth labels y are provided for historical datasets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Canonical feature subsets
X_ONLY_META_FEATURE_NAMES: List[str] = [
    # 1. Dimensionality features (5)
    "n_samples",
    "n_features",
    "feature_to_sample_ratio",
    "log_n_samples",
    "log_n_features",
    # 2. Numeric feature distribution statistics (8)
    "numeric_feature_count",
    "numeric_feature_ratio",
    "mean_of_feature_means",
    "std_of_feature_means",
    "mean_of_feature_stds",
    "std_of_feature_stds",
    "mean_of_feature_min",
    "mean_of_feature_max",
    # 3. Missing value statistics (2)
    "missing_value_ratio",
    "features_with_missing_ratio",
    # 4. Categorical feature statistics (4)
    "categorical_feature_count",
    "categorical_feature_ratio",
    "mean_categorical_cardinality",
    "max_categorical_cardinality",
    # 5. Feature correlation statistics (2)
    "mean_absolute_correlation",
    "max_absolute_correlation",
    # 6. Scale and variability (1)
    "mean_coefficient_of_variation",
]

TARGET_META_FEATURE_NAMES: List[str] = [
    "n_classes",
    "majority_class_ratio",
    "minority_class_ratio",
    "class_entropy",
]

META_FEATURE_NAMES: List[str] = X_ONLY_META_FEATURE_NAMES + TARGET_META_FEATURE_NAMES


@dataclass
class MetaFeatures:
    """Container for dataset-level meta-features.

    Attributes
    ----------
    dataset_name : str
        Name or identifier of the dataset.
    values : dict of str to float or None
        Dictionary mapping meta-feature names to their computed values.
        Target-dependent keys are None when extracted without labels.
    has_target_features : bool
        True if target-dependent features were computed from ground-truth labels.
    """

    dataset_name: str
    values: Dict[str, Optional[float]]
    has_target_features: bool = False


def get_x_only_meta_feature_names() -> List[str]:
    """Return the deterministic list of 22 X-only meta-feature names.

    Returns
    -------
    list of str
        List of 22 X-only meta-feature names in deterministic order.
    """
    return list(X_ONLY_META_FEATURE_NAMES)


def get_target_meta_feature_names() -> List[str]:
    """Return the deterministic list of 4 target-dependent meta-feature names.

    Returns
    -------
    list of str
        List of 4 target-dependent meta-feature names in deterministic order.
    """
    return list(TARGET_META_FEATURE_NAMES)


def get_meta_feature_names(include_target: bool = True) -> List[str]:
    """Return the deterministic list of meta-feature names.

    Parameters
    ----------
    include_target : bool, default=True
        If True, returns all 26 feature names (22 X-only + 4 target-dependent).
        If False, returns only the 22 X-only feature names.

    Returns
    -------
    list of str
        List of meta-feature names.
    """
    if include_target:
        return list(META_FEATURE_NAMES)
    return list(X_ONLY_META_FEATURE_NAMES)


def meta_features_to_array(
    meta_features: MetaFeatures,
    include_target_features: bool = False,
) -> np.ndarray:
    """Convert a MetaFeatures container to a 1D NumPy array with canonical ordering.

    Parameters
    ----------
    meta_features : MetaFeatures
        Container holding extracted meta-features.
    include_target_features : bool, default=False
        If False (default), extracts the 22-dimensional X-only feature vector
        suitable for unseen dataset recommendation without label access.
        If True, extracts the full 26-dimensional feature vector (requires that
        labels were provided during extraction).

    Returns
    -------
    np.ndarray
        1D array of shape (22,) if include_target_features=False, or (26,) if True.

    Raises
    ------
    ValueError
        If include_target_features=True but target features were not computed.
    """
    if include_target_features:
        if not meta_features.has_target_features:
            raise ValueError(
                f"Cannot construct 26-feature vector for '{meta_features.dataset_name}': "
                "target-dependent features are unavailable because extraction was performed with y=None."
            )
        feature_list = META_FEATURE_NAMES
    else:
        feature_list = X_ONLY_META_FEATURE_NAMES

    return np.array(
        [float(meta_features.values[name]) for name in feature_list],
        dtype=np.float64,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, replacing NaN or infinite values with a safe default."""
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def extract_meta_features(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    dataset_name: str = "dataset",
) -> MetaFeatures:
    """Extract dataset-level meta-features from features X and optional target vector y.

    Produces:
    - 22 X-only features when y is None (for unseen dataset recommendations without labels).
    - 26 total features (22 X-only + 4 target-dependent) when y is provided.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_samples, n_features)
        Input feature matrix.
    y : Series or array-like of shape (n_samples,), optional
        Target class labels. If None, target-dependent features are set to None
        and excluded from the standard X-only vector.
    dataset_name : str, default="dataset"
        Identifier for the dataset.

    Returns
    -------
    MetaFeatures
        Dataclass containing dataset name, values dictionary, and has_target_features flag.

    Raises
    ------
    ValueError
        If X is empty or sample counts between X and y mismatch.
    """
    # 1. Convert X to pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    n_samples = len(X_df)
    n_features = X_df.shape[1] if len(X_df.shape) == 2 else 0

    if n_samples == 0:
        raise ValueError(f"Cannot extract meta-features from empty dataset '{dataset_name}'.")
    if n_features == 0:
        raise ValueError(f"Feature matrix for '{dataset_name}' must contain at least one column.")

    # --------------------------------------------------------------------------
    # A. Dimensionality features (5 X-only)
    # --------------------------------------------------------------------------
    feature_to_sample_ratio = n_features / n_samples
    log_n_samples = float(np.log1p(n_samples))
    log_n_features = float(np.log1p(n_features))

    # --------------------------------------------------------------------------
    # B. Missing-value features (2 X-only) & Column partitioning
    # --------------------------------------------------------------------------
    total_cells = n_samples * n_features
    total_missing_cells = int(X_df.isna().sum().sum())
    missing_value_ratio = float(total_missing_cells / total_cells) if total_cells > 0 else 0.0

    cols_with_missing = int((X_df.isna().sum() > 0).sum())
    features_with_missing_ratio = float(cols_with_missing / n_features) if n_features > 0 else 0.0

    # Identify numeric vs categorical columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_df.columns if col not in numeric_cols]

    numeric_feature_count = float(len(numeric_cols))
    numeric_feature_ratio = float(numeric_feature_count / n_features)

    # --------------------------------------------------------------------------
    # C. Numeric-feature distribution statistics (8 X-only) & Scale / Variability (1 X-only)
    # --------------------------------------------------------------------------
    if numeric_cols:
        X_num = X_df[numeric_cols]
        # Feature-level statistics (ddof=0 for population standard deviation)
        feat_means = X_num.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
        feat_stds = X_num.std(axis=0, skipna=True, ddof=0).to_numpy(dtype=np.float64)
        feat_mins = X_num.min(axis=0, skipna=True).to_numpy(dtype=np.float64)
        feat_maxs = X_num.max(axis=0, skipna=True).to_numpy(dtype=np.float64)

        # Replace any residual NaNs (from all-NaN columns) with 0.0
        feat_means = np.nan_to_num(feat_means, nan=0.0)
        feat_stds = np.nan_to_num(feat_stds, nan=0.0)
        feat_mins = np.nan_to_num(feat_mins, nan=0.0)
        feat_maxs = np.nan_to_num(feat_maxs, nan=0.0)

        mean_of_feature_means = float(np.mean(feat_means))
        std_of_feature_means = float(np.std(feat_means, ddof=0))

        mean_of_feature_stds = float(np.mean(feat_stds))
        std_of_feature_stds = float(np.std(feat_stds, ddof=0))

        mean_of_feature_min = float(np.mean(feat_mins))
        mean_of_feature_max = float(np.mean(feat_maxs))

        # Coefficient of variation: CV = abs(std / mean)
        # Avoid division by zero when mean is near zero (|mean| < 1e-12)
        # Cap large finite CVs at 1000.0 to prevent near-zero mean features from dominating
        valid_cvs: List[float] = []
        for m, s in zip(feat_means, feat_stds):
            if abs(m) > 1e-12:
                cv = float(abs(s / m))
                valid_cvs.append(min(cv, 1000.0))

        mean_coefficient_of_variation = float(np.mean(valid_cvs)) if valid_cvs else 0.0

        # Correlation statistics (off-diagonal Pearson correlation)
        if len(numeric_cols) >= 2:
            corr_mat = X_num.corr(method="pearson").to_numpy(dtype=np.float64)
            upper_tri_indices = np.triu_indices_from(corr_mat, k=1)
            off_diag_corrs = np.abs(corr_mat[upper_tri_indices])
            # Ignore NaNs caused by zero-variance constant features
            valid_corrs = off_diag_corrs[~np.isnan(off_diag_corrs)]

            if len(valid_corrs) > 0:
                mean_absolute_correlation = float(np.mean(valid_corrs))
                max_absolute_correlation = float(np.max(valid_corrs))
            else:
                mean_absolute_correlation = 0.0
                max_absolute_correlation = 0.0
        else:
            mean_absolute_correlation = 0.0
            max_absolute_correlation = 0.0
    else:
        mean_of_feature_means = 0.0
        std_of_feature_means = 0.0
        mean_of_feature_stds = 0.0
        std_of_feature_stds = 0.0
        mean_of_feature_min = 0.0
        mean_of_feature_max = 0.0
        mean_coefficient_of_variation = 0.0
        mean_absolute_correlation = 0.0
        max_absolute_correlation = 0.0

    # --------------------------------------------------------------------------
    # D. Categorical-feature statistics (4 X-only)
    # --------------------------------------------------------------------------
    categorical_feature_count = float(len(categorical_cols))
    categorical_feature_ratio = float(categorical_feature_count / n_features)

    if categorical_cols:
        cardinalities = [
            float(X_df[col].dropna().nunique()) for col in categorical_cols
        ]
        mean_categorical_cardinality = float(np.mean(cardinalities)) if cardinalities else 0.0
        max_categorical_cardinality = float(np.max(cardinalities)) if cardinalities else 0.0
    else:
        mean_categorical_cardinality = 0.0
        max_categorical_cardinality = 0.0

    # --------------------------------------------------------------------------
    # E. Target / class features (4 Target-dependent)
    # --------------------------------------------------------------------------
    has_target = y is not None
    if has_target:
        if isinstance(y, pd.DataFrame):
            y_series = y.iloc[:, 0]
        elif not isinstance(y, pd.Series):
            y_series = pd.Series(y)
        else:
            y_series = y.copy()

        if n_samples != len(y_series):
            raise ValueError(
                f"Sample count mismatch in '{dataset_name}': X has {n_samples}, y has {len(y_series)}."
            )

        valid_y = y_series.dropna()
        if len(valid_y) == 0:
            raise ValueError(f"Target vector y for '{dataset_name}' contains only missing values.")

        class_counts = valid_y.value_counts()
        n_classes = float(len(class_counts))
        total_valid_targets = len(valid_y)

        majority_class_ratio = float(class_counts.max() / total_valid_targets)
        minority_class_ratio = float(class_counts.min() / total_valid_targets)

        # Class entropy: H(Y) = -sum(p_i * log2(p_i))
        proportions = class_counts.values / total_valid_targets
        proportions = proportions[proportions > 0]
        class_entropy = float(-np.sum(proportions * np.log2(proportions))) if len(proportions) > 0 else 0.0

        target_values: Dict[str, Optional[float]] = {
            "n_classes": float(n_classes),
            "majority_class_ratio": _safe_float(majority_class_ratio),
            "minority_class_ratio": _safe_float(minority_class_ratio),
            "class_entropy": _safe_float(class_entropy),
        }
    else:
        # Ground-truth labels are deliberately not used
        target_values = {
            "n_classes": None,
            "majority_class_ratio": None,
            "minority_class_ratio": None,
            "class_entropy": None,
        }

    # --------------------------------------------------------------------------
    # Construct ordered values dictionary
    # --------------------------------------------------------------------------
    x_only_values: Dict[str, Optional[float]] = {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "feature_to_sample_ratio": _safe_float(feature_to_sample_ratio),
        "log_n_samples": _safe_float(log_n_samples),
        "log_n_features": _safe_float(log_n_features),
        "numeric_feature_count": _safe_float(numeric_feature_count),
        "numeric_feature_ratio": _safe_float(numeric_feature_ratio),
        "mean_of_feature_means": _safe_float(mean_of_feature_means),
        "std_of_feature_means": _safe_float(std_of_feature_means),
        "mean_of_feature_stds": _safe_float(mean_of_feature_stds),
        "std_of_feature_stds": _safe_float(std_of_feature_stds),
        "mean_of_feature_min": _safe_float(mean_of_feature_min),
        "mean_of_feature_max": _safe_float(mean_of_feature_max),
        "missing_value_ratio": _safe_float(missing_value_ratio),
        "features_with_missing_ratio": _safe_float(features_with_missing_ratio),
        "categorical_feature_count": _safe_float(categorical_feature_count),
        "categorical_feature_ratio": _safe_float(categorical_feature_ratio),
        "mean_categorical_cardinality": _safe_float(mean_categorical_cardinality),
        "max_categorical_cardinality": _safe_float(max_categorical_cardinality),
        "mean_absolute_correlation": _safe_float(mean_absolute_correlation),
        "max_absolute_correlation": _safe_float(max_absolute_correlation),
        "mean_coefficient_of_variation": _safe_float(mean_coefficient_of_variation),
    }

    # Combine in canonical order (X-only first, then target-dependent)
    ordered_values: Dict[str, Optional[float]] = {
        **x_only_values,
        **target_values,
    }

    return MetaFeatures(
        dataset_name=dataset_name,
        values=ordered_values,
        has_target_features=has_target,
    )
