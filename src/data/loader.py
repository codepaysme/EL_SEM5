"""Dataset loading, validation, splitting, and preprocessing for MetaHPO-C."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class LoadedDataset:
    """Container for a loaded and validated dataset.

    Attributes
    ----------
    name : str
        Dataset identifier as listed in configuration.
    X : pd.DataFrame
        Raw feature matrix.
    y : pd.Series
        Target series.
    feature_names : list of str
        List of feature column names.
    target_name : str
        Name of the target variable.
    n_samples : int
        Number of samples (rows).
    n_features : int
        Number of raw input features (columns).
    task_type : str
        Machine learning task type (e.g., 'classification').
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    target_name: str
    n_samples: int
    n_features: int
    task_type: str = "classification"


HISTORICAL_DATASETS: List[str] = [
    "adult",
    "bank-marketing",
    "credit-g",
    "wdbc",
    "spambase",
    "vehicle",
    "phoneme",
]

UNSEEN_DATASETS: List[str] = [
    "madelon",
    "diabetes",
    "steel-plates-fault",
    "ilpd",
]


def get_historical_dataset_names() -> List[str]:
    """Return the canonical list of 7 historical meta-training dataset names.

    Returns
    -------
    list of str
        List of historical dataset identifiers.
    """
    return list(HISTORICAL_DATASETS)


def get_unseen_dataset_names() -> List[str]:
    """Return the canonical list of 4 held-out unseen test dataset names.

    Returns
    -------
    list of str
        List of unseen test dataset identifiers.
    """
    return list(UNSEEN_DATASETS)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns
    -------
    Path
        Absolute path pointing to the project root.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_dataset_config(config_path: Optional[Union[Path, str]] = None) -> List[Dict[str, Any]]:
    """Load and parse dataset configurations from datasets.yaml.

    Parameters
    ----------
    config_path : Path or str, optional
        Path to the YAML configuration file. Defaults to config/datasets.yaml
        relative to the project root.

    Returns
    -------
    list of dict
        List of dataset configuration dictionaries.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration format is invalid.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to parse dataset configurations. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    if config_path is None:
        path = get_project_root() / "config" / "datasets.yaml"
    else:
        path = Path(config_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Dataset configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"Configuration at {path} must contain a top-level 'datasets' list.")

    datasets = data["datasets"]
    if not isinstance(datasets, list):
        raise ValueError("Expected 'datasets' to be a list in datasets.yaml.")

    return datasets


def validate_dataset(
    X: Any,
    y: Any,
    dataset_name: str = "dataset",
) -> None:
    """Validate dataset structure, dimensions, and target properties.

    Parameters
    ----------
    X : Any
        Feature matrix (DataFrame or 2D array).
    y : Any
        Target vector (Series or 1D array).
    dataset_name : str, default="dataset"
        Name of the dataset for descriptive error messages.

    Raises
    ------
    ValueError
        If sample counts mismatch, no features exist, or targets are invalid.
    """
    n_samples_X = len(X)
    n_samples_y = len(y)

    if n_samples_X != n_samples_y:
        raise ValueError(
            f"Sample count mismatch in '{dataset_name}': "
            f"X has {n_samples_X} samples, but y has {n_samples_y} samples."
        )

    if n_samples_X == 0:
        raise ValueError(f"Dataset '{dataset_name}' contains zero samples.")

    if hasattr(X, "shape"):
        if len(X.shape) != 2 or X.shape[1] == 0:
            raise ValueError(f"Feature matrix X for '{dataset_name}' must have at least one feature column.")

    # Target series / array checks
    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    if y_series.isna().all():
        raise ValueError(f"Target vector y for '{dataset_name}' consists entirely of missing values.")

    unique_classes = y_series.dropna().unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset '{dataset_name}' must have at least 2 distinct target classes for classification. "
            f"Found {len(unique_classes)}: {unique_classes.tolist()}"
        )


def load_dataset(
    dataset_name: str,
    config_path: Optional[Union[Path, str]] = None,
) -> LoadedDataset:
    """Load, validate, and wrap a dataset registered in config/datasets.yaml.

    Parameters
    ----------
    dataset_name : str
        The unique identifier of the dataset in datasets.yaml.
    config_path : Path or str, optional
        Path to the dataset configuration file.

    Returns
    -------
    LoadedDataset
        Container containing raw feature DataFrame, target Series, and metadata.

    Raises
    ------
    KeyError
        If dataset_name is not registered in datasets.yaml.
    ValueError
        If the dataset configuration specifies an unsupported source or fails validation.
    RuntimeError
        If fetching the dataset from the remote source fails.
    """
    config_list = load_dataset_config(config_path=config_path)
    entry = next((item for item in config_list if item.get("dataset_name") == dataset_name), None)

    if entry is None:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {[item.get('dataset_name') for item in config_list]}"
        )

    source = entry.get("source", "").lower()
    if source != "openml":
        raise ValueError(
            f"Unsupported source '{source}' for dataset '{dataset_name}'. "
            "Currently only 'openml' source is supported."
        )

    try:
        bunch = fetch_openml(
            name=dataset_name,
            as_frame=True,
            parser="auto",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch dataset '{dataset_name}' from OpenML: {exc}"
        ) from exc

    X = bunch.data
    y = bunch.target

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Ensure y is a 1D Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    elif not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")

    target_name = str(y.name if y.name is not None else "target")
    feature_names = [str(col) for col in X.columns]

    validate_dataset(X, y, dataset_name=dataset_name)

    return LoadedDataset(
        name=dataset_name,
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        n_samples=len(X),
        n_features=X.shape[1],
        task_type="classification",
    )


def split_dataset(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Any, Any, Any]:
    """Create a reproducible stratified train/test split.

    Parameters
    ----------
    X : DataFrame or array of shape (n_samples, n_features)
        Feature matrix.
    y : Series or array of shape (n_samples,)
        Target class labels.
    test_size : float, default=0.2
        Proportion of dataset to include in test split.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)

    Raises
    ------
    ValueError
        If test_size is invalid, dimensions mismatch, or stratification is impossible.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    validate_dataset(X, y, dataset_name="split_dataset_input")

    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    class_counts = y_series.value_counts()
    rare_classes = class_counts[class_counts < 2]

    if not rare_classes.empty:
        raise ValueError(
            f"Cannot perform stratified split: classes {rare_classes.index.tolist()} have fewer than 2 samples."
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_feature_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numeric and categorical feature preprocessing.

    - Numeric columns: Median imputation.
    - Categorical columns: Most frequent imputation followed by OneHotEncoder (unknowns ignored).

    Parameters
    ----------
    X : pd.DataFrame
        Input feature dataframe to inspect column data types.

    Returns
    -------
    ColumnTransformer
        Unfitted scikit-learn ColumnTransformer.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No numeric or categorical columns identified in input DataFrame.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], ColumnTransformer]:
    """Fit feature preprocessor on training data and transform train (and optionally test) features.

    Avoids data leakage by strictly fitting only on X_train.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame, optional
        Test feature matrix.

    Returns
    -------
    tuple
        (X_train_processed, X_test_processed or None, fitted_preprocessor)
    """
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    preprocessor = build_feature_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)

    X_test_proc = None
    if X_test is not None:
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, preprocessor
