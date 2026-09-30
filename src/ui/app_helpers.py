"""Helper functions for the MetaHPO-C Streamlit demonstration application.

Provides clean, reusable utilities for:
- Demo dataset loading (WDBC, Iris, Wine, Breast Cancer)
- CSV upload parsing and validation
- Strict X-only dataset meta-feature extraction (y=None)
- Meta-model loading and error handling
- Candidate generation and confidence-aware recommendation
- Validated research benchmark results loading
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

from src.confidence.acquisition import (
    HPO_SEARCH_SPACE,
    recommend_hyperparameters,
)
from src.data.loader import get_project_root, load_dataset
from src.features.meta_features import (
    X_ONLY_META_FEATURE_NAMES,
    extract_meta_features,
    meta_features_to_array,
)
from src.utils.logging_utils import get_logger

logger = get_logger("ui_app_helpers")

# Canonical 6 Groups for 22 X-Only Meta-Features
META_FEATURE_GROUPS: Dict[str, List[str]] = {
    "Dataset Size": [
        "n_samples",
        "n_features",
        "feature_to_sample_ratio",
        "log_n_samples",
        "log_n_features",
    ],
    "Feature Composition": [
        "numeric_feature_count",
        "numeric_feature_ratio",
        "categorical_feature_count",
        "categorical_feature_ratio",
    ],
    "Distribution Statistics": [
        "mean_of_feature_means",
        "std_of_feature_means",
        "mean_of_feature_stds",
        "std_of_feature_stds",
        "mean_of_feature_min",
        "mean_of_feature_max",
        "mean_coefficient_of_variation",
    ],
    "Missingness": [
        "missing_value_ratio",
        "features_with_missing_ratio",
    ],
    "Categorical Structure": [
        "mean_categorical_cardinality",
        "max_categorical_cardinality",
    ],
    "Correlation": [
        "mean_absolute_correlation",
        "max_absolute_correlation",
    ],
}

# Supported Built-in Demo Datasets
DEMO_DATASETS = ["WDBC", "Iris", "Wine", "Breast Cancer"]


def get_grouped_meta_features(df_meta_table: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Partition the 22 X-only meta-features into their 6 canonical categories.

    Parameters
    ----------
    df_meta_table : pd.DataFrame
        DataFrame with ['Index', 'Meta-Feature', 'Value'].

    Returns
    -------
    dict of {category_name: pd.DataFrame}
    """
    lookup = dict(zip(df_meta_table["Meta-Feature"], df_meta_table["Value"]))
    grouped_tables = {}

    for group_name, feat_names in META_FEATURE_GROUPS.items():
        rows = []
        for feat in feat_names:
            rows.append({
                "Meta-Feature": feat,
                "Value": lookup.get(feat, 0.0),
            })
        grouped_tables[group_name] = pd.DataFrame(rows)

    return grouped_tables


def load_demo_dataset(dataset_name: str) -> Tuple[pd.DataFrame, Optional[pd.Series], str]:
    """Load built-in demonstration dataset.

    Parameters
    ----------
    dataset_name : str
        Name of demo dataset ("WDBC", "Iris", "Wine", "Breast Cancer").

    Returns
    -------
    tuple of (X_df, y_series, display_name)
    """
    clean_name = dataset_name.strip()
    if clean_name == "WDBC":
        try:
            loaded = load_dataset("wdbc")
            return loaded.X, loaded.y, "WDBC (Breast Cancer Diagnostic - OpenML 1510)"
        except Exception:
            ds = load_breast_cancer(as_frame=True)
            return ds.data, ds.target, "WDBC / Breast Cancer Diagnostic (sklearn)"
    elif clean_name == "Iris":
        ds = load_iris(as_frame=True)
        return ds.data, ds.target, "Iris Classification Dataset (150 samples, 4 features)"
    elif clean_name == "Wine":
        ds = load_wine(as_frame=True)
        return ds.data, ds.target, "Wine Recognition Dataset (178 samples, 13 features)"
    elif clean_name == "Breast Cancer":
        ds = load_breast_cancer(as_frame=True)
        return ds.data, ds.target, "Breast Cancer Wisconsin Dataset (569 samples, 30 features)"
    else:
        raise ValueError(f"Unsupported demo dataset '{clean_name}'. Must be one of: {DEMO_DATASETS}")


def parse_uploaded_csv(
    uploaded_file: Any,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Parse and validate user-uploaded CSV file.

    Parameters
    ----------
    uploaded_file : file-like or bytes
        Uploaded CSV file object from Streamlit uploader or file path.
    target_column : str, optional
        Target column name to separate from feature matrix.

    Returns
    -------
    tuple of (X_df, y_series)
    """
    if uploaded_file is None:
        raise ValueError("No file provided.")

    if isinstance(uploaded_file, (str, Path)):
        df = pd.read_csv(uploaded_file)
    elif hasattr(uploaded_file, "read"):
        # Reset stream position if possible
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    elif isinstance(uploaded_file, bytes):
        df = pd.read_csv(io.BytesIO(uploaded_file))
    else:
        raise TypeError(f"Unsupported uploaded file type: {type(uploaded_file)}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    if len(df) < 5:
        raise ValueError(f"Uploaded dataset has too few rows ({len(df)}). At least 5 rows required.")

    if df.shape[1] < 2:
        raise ValueError(f"Uploaded dataset has too few columns ({df.shape[1]}). At least 2 columns required.")

    if target_column and target_column in df.columns:
        X_df = df.drop(columns=[target_column])
        y_series = df[target_column]
    else:
        X_df = df.copy()
        y_series = None

    if X_df.shape[1] == 0:
        raise ValueError("No feature columns remaining after removing target column.")

    return X_df, y_series


def get_dataset_summary_stats(X_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute dataset preview descriptive statistics.

    Parameters
    ----------
    X_df : pd.DataFrame
        Input feature matrix.

    Returns
    -------
    dict
        Dictionary containing summary statistics.
    """
    n_samples, n_features = X_df.shape
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool", "string"]).columns

    total_cells = n_samples * n_features
    missing_cells = int(X_df.isna().sum().sum())
    missing_ratio = float(missing_cells / total_cells) if total_cells > 0 else 0.0

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(cat_cols),
        "missing_value_ratio": missing_ratio,
        "missing_cells_count": missing_cells,
    }


def extract_x_only_meta_features_table(
    X_df: pd.DataFrame,
    dataset_name: str = "target_dataset",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract 22 X-only meta-features (strictly y=None) and format into a display table.

    Parameters
    ----------
    X_df : pd.DataFrame
        Input feature matrix.
    dataset_name : str, default="target_dataset"
        Dataset identifier.

    Returns
    -------
    tuple of (df_meta_table, m_array)
        df_meta_table: DataFrame with columns ['Index', 'Meta-Feature', 'Value']
        m_array: 1D NumPy float64 array of shape (22,)
    """
    # Strict X-only extraction: y is explicitly None
    mf_obj = extract_meta_features(X_df, y=None, dataset_name=dataset_name)
    m_array = meta_features_to_array(mf_obj, include_target_features=False)

    records = []
    for idx, feat_name in enumerate(X_ONLY_META_FEATURE_NAMES, start=1):
        val = mf_obj.values.get(feat_name, 0.0)
        records.append({
            "Index": idx,
            "Meta-Feature": feat_name,
            "Value": float(val) if val is not None else 0.0,
        })

    df_meta_table = pd.DataFrame(records)
    return df_meta_table, m_array


def load_meta_model(model_path: Optional[Union[Path, str]] = None) -> Any:
    """Load the trained ExtraTrees meta-model artifact from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to meta_model.joblib. Defaults to results/experiments/meta_learning/meta_model.joblib.

    Returns
    -------
    Any
        Fitted ExtraTreesRegressor model.
    """
    root = get_project_root()
    p = Path(model_path).resolve() if model_path else root / "results" / "experiments" / "meta_learning" / "meta_model.joblib"

    if not p.exists():
        raise FileNotFoundError(
            f"Trained MetaHPO-C meta-model artifact was not found at: {p}. "
            f"Please verify that Phase 7 training artifacts exist."
        )

    meta_model = joblib.load(p)
    return meta_model


def generate_metahpo_recommendations(
    meta_model: Any,
    dataset_meta_features: np.ndarray,
    n_candidates: int = 200,
    n_select: int = 5,
    beta: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate and rank hyperparameter candidate configurations using MetaHPO-C.

    Parameters
    ----------
    meta_model : Any
        Fitted ExtraTrees meta-model.
    dataset_meta_features : np.ndarray
        22-dimensional X-only meta-feature vector.
    n_candidates : int, default=200
        Size of candidate pool.
    n_select : int, default=5
        Number of top recommendations to return.
    beta : float, default=0.1
        Upper-confidence exploration trade-off parameter.
    seed : int, default=42
        Reproducibility seed for candidate generation.

    Returns
    -------
    pd.DataFrame
        Ranked recommendations table with scores, uncertainties, and hyperparameters.
    """
    df_rec = recommend_hyperparameters(
        meta_model=meta_model,
        dataset_meta_features=dataset_meta_features,
        n_candidates=n_candidates,
        n_select=n_select,
        beta=beta,
        seed=seed,
    )

    # Format column names for presentation
    df_formatted = df_rec.copy()
    df_formatted["Rank"] = np.arange(1, len(df_formatted) + 1)
    if "predicted_score" in df_formatted.columns and "predicted_cv_macro_f1" not in df_formatted.columns:
        df_formatted["predicted_cv_macro_f1"] = df_formatted["predicted_score"]

    # Reorder columns logically
    metric_cols = ["Rank", "predicted_cv_macro_f1", "prediction_uncertainty", "relative_confidence", "acquisition_score"]
    param_cols = list(HPO_SEARCH_SPACE.keys())
    available_cols = [c for c in metric_cols if c in df_formatted.columns] + [c for c in param_cols if c in df_formatted.columns]

    df_formatted = df_formatted[available_cols].reset_index(drop=True)
    return df_formatted



def load_validated_benchmark_results(results_path: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Load previously validated Phase 10 & 11 benchmark results table.

    Parameters
    ----------
    results_path : Path or str, optional
        Path to final_results_table.csv.

    Returns
    -------
    pd.DataFrame
        Benchmark summary table.
    """
    root = get_project_root()
    p1 = Path(results_path).resolve() if results_path else root / "results" / "analysis" / "final_results_table.csv"
    p2 = root / "results" / "experiments" / "unseen_benchmark" / "unseen_benchmark_summary.csv"

    if p1.exists():
        return pd.read_csv(p1)
    elif p2.exists():
        return pd.read_csv(p2)
    else:
        raise FileNotFoundError("Benchmark results artifacts not found.")
