"""Prediction uncertainty and relative confidence estimation for MetaHPO-C.

Conceptual Architecture & Terminology:
--------------------------------------
In MetaHPO-C, confidence estimation is grounded in the ensemble prediction dispersion
of the ExtraTrees response-surface surrogate:

    u(x) = std(p_1(x), p_2(x), ..., p_T(x))

Terminology & Methodological Distinctions:
1. "prediction uncertainty proxy" or "ensemble-dispersion uncertainty proxy":
   The standard deviation of predictions across the T=300 individual decision trees.
   - Low dispersion indicates strong tree agreement.
   - High dispersion indicates strong tree disagreement.
2. "relative confidence heuristic":
   A bounded score in (0, 1] derived from normalized dispersion:
       relative_confidence = 1 / (1 + u(x) / (median(u) + eps))
3. CALIBRATION NOTICE:
   This quantity is an empirical heuristic for ranking and filtering candidate
   hyperparameter configurations. It is NOT a formally calibrated posterior probability
   or Bayesian credible interval (e.g., confidence 0.8 does NOT mean "80% probability").
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.data.loader import HISTORICAL_DATASETS, get_project_root
from src.meta_learning.meta_dataset import (
    META_INPUT_FEATURE_NAMES,
    META_TARGET_NAME,
)
from src.meta_learning.meta_model import (
    DEFAULT_EXTRA_TREES_PARAMS,
    train_meta_model,
)
from src.utils.logging_utils import get_logger

logger = get_logger("uncertainty")


@dataclass
class UncertaintyPrediction:
    """Container for ensemble predictions and dispersion uncertainty estimates.

    Attributes
    ----------
    mean : np.ndarray
        Ensemble mean predicted target values (shape: n_samples).
    std : np.ndarray
        Ensemble standard deviation (uncertainty proxy) across trees (shape: n_samples).
    tree_predictions : np.ndarray, optional
        Raw prediction matrix from all individual trees (shape: n_trees, n_samples).
    """

    mean: np.ndarray
    std: np.ndarray
    tree_predictions: Optional[np.ndarray] = None


def predict_with_uncertainty(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    return_tree_predictions: bool = False,
    ddof: int = 0,
) -> UncertaintyPrediction:
    """Compute ensemble mean and dispersion-based uncertainty proxy from tree predictions.

    Parameters
    ----------
    model : fitted ExtraTreesRegressor or RandomForestRegressor
        Fitted ensemble model containing the 'estimators_' attribute.
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Predictive feature matrix.
    return_tree_predictions : bool, default=False
        Whether to include the full (n_trees, n_samples) matrix in the returned container.
    ddof : int, default=0
        Delta degrees of freedom for standard deviation calculation (0 = population std).

    Returns
    -------
    UncertaintyPrediction
        Container with mean and std (uncertainty proxy) arrays.

    Raises
    ------
    AttributeError
        If model does not possess an 'estimators_' attribute.
    """
    if not hasattr(model, "estimators_"):
        raise AttributeError("Model does not have 'estimators_' attribute. Expected a tree ensemble regressor.")

    # Convert to ndarray if DataFrame to avoid per-tree feature name validation warnings
    X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

    # Matrix of shape (n_estimators, n_samples)
    tree_preds = np.vstack([
        tree.predict(X_mat)
        for tree in model.estimators_
    ])

    mean_pred = np.mean(tree_preds, axis=0)
    std_pred = np.std(tree_preds, axis=0, ddof=ddof)

    return UncertaintyPrediction(
        mean=np.asarray(mean_pred, dtype=np.float64),
        std=np.asarray(std_pred, dtype=np.float64),
        tree_predictions=tree_preds if return_tree_predictions else None,
    )


def estimate_confidence(
    uncertainty: Union[np.ndarray, pd.Series, float, List[float]],
    reference_uncertainties: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
    method: str = "relative",
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Transform prediction uncertainty into a bounded relative confidence score in (0, 1].

    NOTE: This is an empirical relative confidence heuristic for comparing candidate
    configurations and is NOT a statistically calibrated probability.

    Parameters
    ----------
    uncertainty : array-like or float
        Uncertainty proxy value(s) (non-negative standard deviation).
    reference_uncertainties : array-like, optional
        Reference set of uncertainties to establish scaling median. If None, uses
        the median of the input uncertainty array.
    method : str, default='relative'
        Transformation method ('relative' or 'absolute'):
        - 'relative': confidence = 1 / (1 + u / (median(u_ref) + eps))
        - 'absolute': confidence = 1 / (1 + u)
    epsilon : float, default=1e-12
        Small numerical stability constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Relative confidence scores in the range (0, 1].

    Raises
    ------
    ValueError
        If any uncertainty values are negative.
    """
    u = np.asarray(uncertainty, dtype=np.float64)

    if np.any(u < 0):
        raise ValueError("Uncertainty values must be non-negative.")

    if method == "absolute":
        return 1.0 / (1.0 + u)

    if reference_uncertainties is not None:
        ref_u = np.asarray(reference_uncertainties, dtype=np.float64)
        median_u = float(np.median(ref_u))
    else:
        median_u = float(np.median(u)) if u.size > 0 else 1.0

    normalized_u = u / (median_u + epsilon)
    return 1.0 / (1.0 + normalized_u)


@dataclass
class LODOUncertaintyResult:
    """Container for Leave-One-Dataset-Out (LODO) uncertainty evaluation results."""

    predictions_df: pd.DataFrame
    metrics: Dict[str, Any]
    per_dataset_metrics: Dict[str, Dict[str, Any]]

    def save(
        self,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Tuple[Path, Path]:
        """Save lodo_uncertainty_predictions.csv and uncertainty_metrics.json.

        Parameters
        ----------
        output_dir : Path or str, optional
            Destination directory. Defaults to results/experiments/meta_learning.

        Returns
        -------
        tuple of (Path, Path)
            Paths to (lodo_uncertainty_predictions.csv, uncertainty_metrics.json).
        """
        if output_dir is None:
            out_path = get_project_root() / "results" / "experiments" / "meta_learning"
        else:
            out_path = Path(output_dir).resolve()

        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Save predictions CSV
        csv_path = out_path / "lodo_uncertainty_predictions.csv"
        self.predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved LODO uncertainty predictions ({len(self.predictions_df)} rows) to: {csv_path.name}")

        # 2. Save metrics JSON
        json_path = out_path / "uncertainty_metrics.json"
        full_metrics = {
            "validation_method": "Leave-One-Dataset-Out (LODO)",
            "uncertainty_method": "ExtraTrees ensemble prediction dispersion",
            "n_estimators": self.metrics.get("n_estimators", 300),
            "uncertainty_definition": "Standard deviation across tree predictions p_1..p_T",
            "standard_deviation_ddof": 0,
            "calibrated": False,
            "interpretation": "ensemble disagreement proxy (relative heuristic, not calibrated probability)",
            "overall_mean_uncertainty": float(self.metrics["overall_mean_uncertainty"]),
            "overall_median_uncertainty": float(self.metrics["overall_median_uncertainty"]),
            "overall_mean_absolute_error": float(self.metrics["overall_mean_absolute_error"]),
            "overall_median_absolute_error": float(self.metrics["overall_median_absolute_error"]),
            "overall_spearman_uncertainty_vs_absolute_error": float(self.metrics["overall_spearman_uncertainty_vs_absolute_error"]),
            "overall_pearson_uncertainty_vs_absolute_error": float(self.metrics["overall_pearson_uncertainty_vs_absolute_error"]),
            "low_uncertainty_mae": float(self.metrics["low_uncertainty_mae"]),
            "high_uncertainty_mae": float(self.metrics["high_uncertainty_mae"]),
            "per_dataset_metrics": self.per_dataset_metrics,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_metrics, f, indent=2)
        logger.info(f"Saved uncertainty evaluation metrics to: {json_path.name}")

        return csv_path, json_path


def evaluate_lodo_uncertainty(
    meta_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    target_name: str = META_TARGET_NAME,
    model_params: Optional[Dict[str, Any]] = None,
) -> LODOUncertaintyResult:
    """Evaluate ensemble uncertainty estimation under Leave-One-Dataset-Out cross-validation.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Complete meta-learning dataset DataFrame (140 rows).
    feature_names : list of str, optional
        Predictive feature names (30 features). Defaults to META_INPUT_FEATURE_NAMES.
    target_name : str, default='cv_macro_f1'
        Name of target column.
    model_params : dict, optional
        Custom ExtraTreesRegressor parameters.

    Returns
    -------
    LODOUncertaintyResult
        Structured container with predictions DataFrame and correlation/diagnostic metrics.
    """
    if feature_names is None:
        feature_names = list(META_INPUT_FEATURE_NAMES)

    if "dataset_name" not in meta_df.columns:
        raise ValueError("Missing 'dataset_name' column required for LODO uncertainty evaluation.")

    unique_datasets = sorted(meta_df["dataset_name"].unique())
    logger.info(f"Starting LODO uncertainty evaluation across {len(unique_datasets)} datasets: {unique_datasets}")

    all_rows: List[Dict[str, Any]] = []
    per_dataset_metrics: Dict[str, Dict[str, Any]] = {}

    for ds in unique_datasets:
        train_mask = meta_df["dataset_name"] != ds
        val_mask = meta_df["dataset_name"] == ds

        train_df = meta_df[train_mask]
        val_df = meta_df[val_mask]

        X_train = train_df[feature_names]
        y_train = train_df[target_name]

        X_val = val_df[feature_names]
        y_val = val_df[target_name].to_numpy()

        # Fit ExtraTrees meta-model on the 6 training datasets
        fold_model = train_meta_model(X_train, y_train, model_params=model_params)

        # Extract predictions and dispersion uncertainty
        unc_pred = predict_with_uncertainty(fold_model, X_val, ddof=0)
        y_pred = unc_pred.mean
        y_unc = unc_pred.std
        abs_err = np.abs(y_val - y_pred)

        # Compute relative confidence for the fold
        rel_conf = estimate_confidence(y_unc, method="relative")

        val_trial_nums = val_df["trial_number"].tolist() if "trial_number" in val_df.columns else list(range(len(val_df)))
        for trial_num, act, pred, unc, err, conf in zip(val_trial_nums, y_val, y_pred, y_unc, abs_err, rel_conf):
            all_rows.append({
                "dataset_name": ds,
                "trial_number": int(trial_num),
                "actual_cv_macro_f1": float(act),
                "predicted_cv_macro_f1": float(pred),
                "prediction_uncertainty": float(unc),
                "absolute_error": float(err),
                "relative_confidence": float(conf),
            })

        # Compute fold-level correlations
        if np.std(y_unc) > 0 and np.std(abs_err) > 0:
            sp_corr, _ = spearmanr(y_unc, abs_err)
            pr_corr, _ = pearsonr(y_unc, abs_err)
            sp_val = 0.0 if np.isnan(sp_corr) else float(sp_corr)
            pr_val = 0.0 if np.isnan(pr_corr) else float(pr_corr)
        else:
            sp_val, pr_val = 0.0, 0.0

        # Fold-level low vs high uncertainty split
        med_unc = np.median(y_unc)
        low_mask = y_unc <= med_unc
        high_mask = y_unc > med_unc

        fold_low_mae = float(np.mean(abs_err[low_mask])) if np.any(low_mask) else float(np.mean(abs_err))
        fold_high_mae = float(np.mean(abs_err[high_mask])) if np.any(high_mask) else float(np.mean(abs_err))

        per_dataset_metrics[ds] = {
            "mean_uncertainty": float(np.mean(y_unc)),
            "median_uncertainty": float(np.median(y_unc)),
            "mean_absolute_error": float(np.mean(abs_err)),
            "median_absolute_error": float(np.median(abs_err)),
            "spearman_uncertainty_vs_error": sp_val,
            "pearson_uncertainty_vs_error": pr_val,
            "low_uncertainty_mae": fold_low_mae,
            "high_uncertainty_mae": fold_high_mae,
        }

    predictions_df = pd.DataFrame(all_rows)

    # Global aggregate metrics across all 140 predictions
    all_unc = predictions_df["prediction_uncertainty"].to_numpy()
    all_err = predictions_df["absolute_error"].to_numpy()

    if np.std(all_unc) > 0 and np.std(all_err) > 0:
        overall_sp, _ = spearmanr(all_unc, all_err)
        overall_pr, _ = pearsonr(all_unc, all_err)
        overall_sp_val = 0.0 if np.isnan(overall_sp) else float(overall_sp)
        overall_pr_val = 0.0 if np.isnan(overall_pr) else float(overall_pr)
    else:
        overall_sp_val, overall_pr_val = 0.0, 0.0

    # Overall diagnostic: split all 140 predictions into low/high uncertainty halves
    global_med_unc = np.median(all_unc)
    low_unc_mask = all_unc <= global_med_unc
    high_unc_mask = all_unc > global_med_unc

    overall_low_mae = float(np.mean(all_err[low_unc_mask]))
    overall_high_mae = float(np.mean(all_err[high_unc_mask]))

    effective_params = model_params or DEFAULT_EXTRA_TREES_PARAMS

    overall_metrics = {
        "n_estimators": effective_params.get("n_estimators", 300),
        "overall_mean_uncertainty": float(np.mean(all_unc)),
        "overall_median_uncertainty": float(global_med_unc),
        "overall_mean_absolute_error": float(np.mean(all_err)),
        "overall_median_absolute_error": float(np.median(all_err)),
        "overall_spearman_uncertainty_vs_absolute_error": overall_sp_val,
        "overall_pearson_uncertainty_vs_absolute_error": overall_pr_val,
        "low_uncertainty_mae": overall_low_mae,
        "high_uncertainty_mae": overall_high_mae,
    }

    return LODOUncertaintyResult(
        predictions_df=predictions_df,
        metrics=overall_metrics,
        per_dataset_metrics=per_dataset_metrics,
    )
