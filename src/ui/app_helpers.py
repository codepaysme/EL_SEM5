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
from src.confidence.pruning import (
    compute_space_volume_ratio,
    prune_search_space,
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

# Supported Built-in Demo Datasets (including Indian datasets & real-world benchmarks)
DEMO_DATASETS = [
    "WDBC (Breast Cancer Diagnostic)",
    "ILPD (Indian Liver Patient - AP, India)",
    "Diabetes (Pima Indians Clinical)",
    "Steel Plates Fault (Industrial Quality)",
    "Credit-G (German Credit Risk)",
    "Spambase (Email Spam Detection)",
    "Vehicle (Silhouette Recognition)",
    "Phoneme (Acoustic Speech Signals)",
    "Iris (Botanical Classification)",
    "Wine (Chemical Composition)",
]

# Aliases for backward-compatibility with test suites and short names
DEMO_DATASET_ALIASES: Dict[str, str] = {
    "WDBC": "WDBC (Breast Cancer Diagnostic)",
    "Breast Cancer": "WDBC (Breast Cancer Diagnostic)",
    "Iris": "Iris (Botanical Classification)",
    "Wine": "Wine (Chemical Composition)",
    "ILPD": "ILPD (Indian Liver Patient - AP, India)",
    "Diabetes": "Diabetes (Pima Indians Clinical)",
    "Steel Plates Fault": "Steel Plates Fault (Industrial Quality)",
    "Credit-G": "Credit-G (German Credit Risk)",
    "Spambase": "Spambase (Email Spam Detection)",
    "Vehicle": "Vehicle (Silhouette Recognition)",
    "Phoneme": "Phoneme (Acoustic Speech Signals)",
}

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "WDBC (Breast Cancer Diagnostic)": {
        "cache_file": "wdbc.csv",
        "openml_name": "wdbc",
        "display_name": "WDBC (Breast Cancer Diagnostic - OpenML 1510 / Sklearn)",
        "sklearn_loader": load_breast_cancer,
    },
    "ILPD (Indian Liver Patient - AP, India)": {
        "cache_file": "ilpd.csv",
        "openml_name": "ilpd",
        "display_name": "ILPD (Indian Liver Patient Dataset - Andhra Pradesh, India)",
        "sklearn_loader": None,
    },
    "Diabetes (Pima Indians Clinical)": {
        "cache_file": "diabetes.csv",
        "openml_name": "diabetes",
        "display_name": "Pima Indians Diabetes Clinical Dataset (768 samples, 8 features)",
        "sklearn_loader": None,
    },
    "Steel Plates Fault (Industrial Quality)": {
        "cache_file": "steel_plates_fault.csv",
        "openml_name": "steel-plates-fault",
        "display_name": "Steel Plates Fault Manufacturing Dataset (1,941 samples, 33 features)",
        "sklearn_loader": None,
    },
    "Credit-G (German Credit Risk)": {
        "cache_file": "credit_g.csv",
        "openml_name": "credit-g",
        "display_name": "Credit-G Financial Risk Classification Dataset (1,000 samples, 20 features)",
        "sklearn_loader": None,
    },
    "Spambase (Email Spam Detection)": {
        "cache_file": "spambase.csv",
        "openml_name": "spambase",
        "display_name": "Spambase Text Frequency Classification Dataset (4,601 samples, 57 features)",
        "sklearn_loader": None,
    },
    "Vehicle (Silhouette Recognition)": {
        "cache_file": "vehicle.csv",
        "openml_name": "vehicle",
        "display_name": "Vehicle 3D Silhouette Recognition Dataset (846 samples, 18 features)",
        "sklearn_loader": None,
    },
    "Phoneme (Acoustic Speech Signals)": {
        "cache_file": "phoneme.csv",
        "openml_name": "phoneme",
        "display_name": "Phoneme Acoustic Speech Signal Dataset (5,404 samples, 5 features)",
        "sklearn_loader": None,
    },
    "Iris (Botanical Classification)": {
        "cache_file": "iris.csv",
        "openml_name": None,
        "display_name": "Iris Classification Dataset (150 samples, 4 features)",
        "sklearn_loader": load_iris,
    },
    "Wine (Chemical Composition)": {
        "cache_file": "wine.csv",
        "openml_name": None,
        "display_name": "Wine Recognition Dataset (178 samples, 13 features)",
        "sklearn_loader": load_wine,
    },
}


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
    """Load built-in demonstration dataset (supports Indian datasets, OpenML, & Sklearn).

    Parameters
    ----------
    dataset_name : str
        Name or identifier of demo dataset.

    Returns
    -------
    tuple of (X_df, y_series, display_name)
    """
    clean_name = dataset_name.strip()
    canonical_name = DEMO_DATASET_ALIASES.get(clean_name, clean_name)

    if canonical_name not in DATASET_METADATA:
        valid_choices = DEMO_DATASETS + list(DEMO_DATASET_ALIASES.keys())
        raise ValueError(f"Unsupported demo dataset '{clean_name}'. Must be one of: {valid_choices}")

    meta = DATASET_METADATA[canonical_name]
    cache_path = get_project_root() / "data" / "demo_cache" / meta["cache_file"]

    # 1. Preferred fast, deterministic, 100% offline cache
    if cache_path.is_file():
        try:
            df = pd.read_csv(cache_path)
            if "target" in df.columns:
                X_df = df.drop(columns=["target"])
                y_series = df["target"]
            else:
                X_df = df.iloc[:, :-1]
                y_series = df.iloc[:, -1]
            return X_df, y_series, meta["display_name"]
        except Exception as exc:
            logger.warning(f"Failed loading cached CSV at {cache_path}: {exc}. Falling back to dynamic load.")

    # 2. Sklearn built-in fallback
    if meta.get("sklearn_loader") is not None:
        ds = meta["sklearn_loader"](as_frame=True)
        return ds.data, ds.target, meta["display_name"]

    # 3. OpenML remote/cached fallback
    if meta.get("openml_name") is not None:
        loaded = load_dataset(meta["openml_name"])
        return loaded.X, loaded.y, meta["display_name"]

    raise RuntimeError(f"Unable to load demo dataset '{dataset_name}' from cache or remote repositories.")


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


def format_search_space_comparison(
    base_space: Dict[str, Tuple[Union[int, float], Union[int, float], str]],
    pruned_space: Dict[str, Tuple[Union[int, float], Union[int, float], str]],
) -> pd.DataFrame:
    """Format a side-by-side comparison table between original and pruned search spaces.

    Parameters
    ----------
    base_space : dict
        Baseline search space bounds.
    pruned_space : dict
        Pruned search space bounds.

    Returns
    -------
    pd.DataFrame
        Table with ['Hyperparameter', 'Original Search Space', 'Pruned Search Space', 'Status'].
    """
    rows = []
    for param, (b_low, b_high, ptype) in base_space.items():
        p_low, p_high, _ = pruned_space[param]
        is_pruned = (b_low != p_low) or (b_high != p_high)

        if ptype == "int":
            orig_str = f"[{int(b_low)}, {int(b_high)}]"
            prun_str = f"[{int(p_low)}, {int(p_high)}]"
        elif ptype == "log_float":
            orig_str = f"[{b_low:.4f}, {b_high:.4f}] (log)"
            prun_str = f"[{p_low:.4f}, {p_high:.4f}] (log)"
        else:
            orig_str = f"[{b_low:.4f}, {b_high:.4f}]"
            prun_str = f"[{p_low:.4f}, {p_high:.4f}]"

        status_str = "🎯 Contracted" if is_pruned else "— Unchanged"
        rows.append({
            "Hyperparameter": param,
            "Original Search Space": orig_str,
            "Pruned Search Space": prun_str,
            "Status": status_str,
        })

    return pd.DataFrame(rows)


def generate_metahpo_recommendations(
    meta_model: Any,
    dataset_meta_features: np.ndarray,
    n_candidates: int = 200,
    n_select: int = 5,
    beta: float = 0.1,
    search_space: Optional[Dict[str, Tuple[Union[int, float], Union[int, float], str]]] = None,
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
    search_space : dict, optional
        Custom or pruned search space dictionary. Defaults to canonical HPO_SEARCH_SPACE.
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
        search_space=search_space,
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
