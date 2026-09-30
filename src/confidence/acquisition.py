"""Confidence-aware hyperparameter candidate selection for MetaHPO-C.

Conceptual Architecture & Methodological Interpretation:
---------------------------------------------------------
In MetaHPO-C, candidate hyperparameter selection bridges the response-surface
meta-learner with the downstream optimization engine:

1. Candidate Generation:
   Uniform and log-uniform random sampling of candidate configurations theta within
   the compact 8-parameter XGBoost search space.

2. Surrogate Evaluation & Uncertainty Estimation:
   The candidate feature matrix [m, theta] is evaluated by the ExtraTrees meta-model:
   - predicted_score = mean(p_1(x), ..., p_T(x))
   - prediction_uncertainty = std(p_1(x), ..., p_T(x))   [Phase 8 ensemble dispersion]
   - relative_confidence = 1 / (1 + u / (median(u) + eps))

3. Confidence-Aware Acquisition Score:
   Candidates are ranked using an upper-confidence-style acquisition heuristic:
       acquisition_score = predicted_score + beta * prediction_uncertainty

IMPORTANT METHODOLOGICAL NOTICE:
This is an empirical, heuristic acquisition rule designed to balance exploitation
(high predicted validation score) and exploration (high tree-disagreement dispersion).
It is NOT Bayesian optimization, does NOT represent a calibrated posterior probability,
and does NOT constitute a formal Bayesian acquisition function (e.g., GP-UCB / Expected
Improvement). The uncertainty quantity is strictly an ensemble-disagreement proxy.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.confidence.uncertainty import estimate_confidence, predict_with_uncertainty
from src.features.meta_features import X_ONLY_META_FEATURE_NAMES
from src.meta_learning.meta_dataset import HPO_PARAMETER_NAMES, META_INPUT_FEATURE_NAMES
from src.utils.logging_utils import get_logger

logger = get_logger("acquisition")

# Canonical search space bounds matching Phase 4/5 HPO optimizer
HPO_SEARCH_SPACE: Dict[str, Tuple[Union[int, float], Union[int, float], str]] = {
    "n_estimators": (50, 300, "int"),
    "max_depth": (2, 8, "int"),
    "learning_rate": (0.01, 0.3, "log_float"),
    "subsample": (0.6, 1.0, "float"),
    "colsample_bytree": (0.6, 1.0, "float"),
    "min_child_weight": (1, 10, "int"),
    "gamma": (0.0, 2.0, "float"),
    "reg_lambda": (0.1, 10.0, "log_float"),
}


def generate_candidate_configs(
    n_candidates: int = 200,
    seed: int = 42,
) -> List[Dict[str, Union[float, int]]]:
    """Sample candidate XGBoost hyperparameter configurations from the search space.

    Parameters
    ----------
    n_candidates : int, default=200
        Number of candidate hyperparameter configurations to generate.
    seed : int, default=42
        Reproducibility seed for NumPy random number generator.

    Returns
    -------
    list of dict
        List of candidate configuration dictionaries containing all 8 hyperparameters.

    Raises
    ------
    ValueError
        If n_candidates <= 0.
    """
    if n_candidates <= 0:
        raise ValueError(f"n_candidates must be a positive integer, got {n_candidates}.")

    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Union[float, int]]] = []

    for _ in range(n_candidates):
        config: Dict[str, Union[float, int]] = {}
        for param, (low, high, ptype) in HPO_SEARCH_SPACE.items():
            if ptype == "int":
                config[param] = int(rng.integers(int(low), int(high) + 1))
            elif ptype == "float":
                config[param] = float(rng.uniform(float(low), float(high)))
            elif ptype == "log_float":
                log_low = np.log(float(low))
                log_high = np.log(float(high))
                config[param] = float(np.exp(rng.uniform(log_low, log_high)))
            else:
                raise ValueError(f"Unknown parameter type '{ptype}' for {param}.")
        candidates.append(config)

    return candidates


def build_candidate_matrix(
    dataset_meta_features: Union[np.ndarray, List[float], pd.Series],
    candidate_configs: List[Dict[str, Union[float, int]]],
) -> np.ndarray:
    """Construct 30-dimensional predictive feature matrix [m, theta] for all candidate configs.

    Parameters
    ----------
    dataset_meta_features : array-like of shape (22,)
        Canonical 22 X-only meta-features for the target dataset.
    candidate_configs : list of dict
        Candidate hyperparameter dictionaries (each containing the 8 hyperparameters).

    Returns
    -------
    np.ndarray of shape (n_candidates, 30)
        Constructed candidate feature matrix ordered as [22 X-only meta-features, 8 hyperparameters].

    Raises
    ------
    ValueError
        If meta-features count != 22, candidate_configs is empty, or inputs contain NaNs/infinities.
    """
    m = np.asarray(dataset_meta_features, dtype=np.float64).flatten()

    if len(m) != len(X_ONLY_META_FEATURE_NAMES):
        raise ValueError(
            f"Expected exactly {len(X_ONLY_META_FEATURE_NAMES)} X-only meta-features, got {len(m)}."
        )
    if not np.all(np.isfinite(m)):
        raise ValueError("dataset_meta_features contains non-finite values (inf or nan).")

    if not candidate_configs:
        raise ValueError("candidate_configs list cannot be empty.")

    n_candidates = len(candidate_configs)
    matrix = np.zeros((n_candidates, len(META_INPUT_FEATURE_NAMES)), dtype=np.float64)

    for i, cfg in enumerate(candidate_configs):
        # 1. First 22 columns: identical dataset meta-features
        matrix[i, : len(X_ONLY_META_FEATURE_NAMES)] = m

        # 2. Next 8 columns: candidate hyperparameters in canonical order
        for j, param_name in enumerate(HPO_PARAMETER_NAMES):
            if param_name not in cfg:
                raise ValueError(f"Candidate configuration {i} is missing required parameter '{param_name}'.")
            val = float(cfg[param_name])
            if not np.isfinite(val):
                raise ValueError(f"Candidate configuration {i} has non-finite value for '{param_name}'.")
            matrix[i, len(X_ONLY_META_FEATURE_NAMES) + j] = val

    return matrix


def score_candidates(
    meta_model: Any,
    candidate_matrix: np.ndarray,
    candidate_configs: Optional[List[Dict[str, Union[float, int]]]] = None,
    reference_uncertainties: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate candidate configurations using the meta-model, predicting scores and uncertainty.

    Parameters
    ----------
    meta_model : fitted ExtraTreesRegressor or MetaModel
        Fitted response-surface meta-learner with 'estimators_' attribute.
    candidate_matrix : np.ndarray of shape (n_candidates, 30)
        30-dimensional predictive matrix.
    candidate_configs : list of dict, optional
        Original candidate configuration dictionaries. If None, extracted from candidate_matrix.
    reference_uncertainties : np.ndarray, optional
        Reference uncertainty distribution for relative confidence scaling.

    Returns
    -------
    pd.DataFrame
        DataFrame containing candidate_id, 8 hyperparameters, predicted_score,
        prediction_uncertainty, and relative_confidence.
    """
    if hasattr(meta_model, "model") and meta_model.model is not None:
        underlying_model = meta_model.model
    else:
        underlying_model = meta_model

    # Extract ensemble predictions and dispersion uncertainty
    unc_res = predict_with_uncertainty(underlying_model, candidate_matrix, ddof=0)
    pred_scores = unc_res.mean
    pred_unc = unc_res.std

    # Compute relative confidence
    rel_conf = estimate_confidence(
        pred_unc,
        reference_uncertainties=reference_uncertainties,
        method="relative",
    )

    n_candidates = candidate_matrix.shape[0]
    rows: List[Dict[str, Any]] = []

    for i in range(n_candidates):
        row: Dict[str, Any] = {"candidate_id": i}

        # Hyperparameter values
        if candidate_configs is not None and i < len(candidate_configs):
            for p in HPO_PARAMETER_NAMES:
                row[p] = candidate_configs[i][p]
        else:
            for j, p in enumerate(HPO_PARAMETER_NAMES):
                val = candidate_matrix[i, len(X_ONLY_META_FEATURE_NAMES) + j]
                ptype = HPO_SEARCH_SPACE[p][2]
                row[p] = int(round(val)) if ptype == "int" else float(val)

        row["predicted_score"] = float(pred_scores[i])
        row["prediction_uncertainty"] = float(pred_unc[i])
        row["relative_confidence"] = float(rel_conf[i])
        rows.append(row)

    cols = ["candidate_id"] + HPO_PARAMETER_NAMES + [
        "predicted_score",
        "prediction_uncertainty",
        "relative_confidence",
    ]
    return pd.DataFrame(rows)[cols]


def compute_acquisition_score(
    predicted_score: Union[np.ndarray, pd.Series, List[float]],
    uncertainty: Union[np.ndarray, pd.Series, List[float]],
    beta: float = 0.1,
) -> np.ndarray:
    """Calculate upper-confidence-style heuristic acquisition score.

    acquisition = predicted_score + beta * uncertainty

    Parameters
    ----------
    predicted_score : array-like of shape (n_candidates,)
        Expected validation CV macro F1 from meta-model.
    uncertainty : array-like of shape (n_candidates,)
        Ensemble prediction dispersion (standard deviation).
    beta : float, default=0.1
        Exploration weighting parameter (must be non-negative).

    Returns
    -------
    np.ndarray of shape (n_candidates,)
        Acquisition scores.

    Raises
    ------
    ValueError
        If beta < 0, shapes mismatch, or inputs contain non-finite values.
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative (>= 0), got {beta}.")

    p = np.asarray(predicted_score, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)

    if p.shape != u.shape:
        raise ValueError(f"Shape mismatch: predicted_score shape {p.shape} != uncertainty shape {u.shape}.")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(u)):
        raise ValueError("predicted_score and uncertainty must contain only finite values.")
    if np.any(u < 0):
        raise ValueError("uncertainty values must be non-negative.")

    return p + (beta * u)


def select_top_candidates(
    scored_candidates: pd.DataFrame,
    n_select: int = 10,
    beta: float = 0.1,
) -> pd.DataFrame:
    """Score, rank, and select the top candidate configurations using confidence-aware acquisition.

    Parameters
    ----------
    scored_candidates : pd.DataFrame
        DataFrame produced by score_candidates, containing predicted_score,
        prediction_uncertainty, candidate_id, and hyperparameter columns.
    n_select : int, default=10
        Number of top candidate configurations to return.
    beta : float, default=0.1
        Exploration weighting parameter.

    Returns
    -------
    pd.DataFrame
        Top n_select candidate configurations sorted descending by acquisition_score.
    """
    if n_select <= 0:
        raise ValueError(f"n_select must be a positive integer, got {n_select}.")
    if "predicted_score" not in scored_candidates.columns or "prediction_uncertainty" not in scored_candidates.columns:
        raise ValueError("scored_candidates DataFrame must contain 'predicted_score' and 'prediction_uncertainty'.")

    df = scored_candidates.copy()

    # Calculate acquisition score
    acq = compute_acquisition_score(
        df["predicted_score"].to_numpy(),
        df["prediction_uncertainty"].to_numpy(),
        beta=beta,
    )
    df["acquisition_score"] = acq

    # Deterministic sorting: descending by acquisition_score, ascending by candidate_id for ties
    sort_cols = ["acquisition_score", "candidate_id"]
    sort_ascending = [False, True]
    df_sorted = df.sort_values(by=sort_cols, ascending=sort_ascending).reset_index(drop=True)

    # Return top n_select rows
    return df_sorted.iloc[: min(n_select, len(df_sorted))].copy()


def recommend_hyperparameters(
    meta_model: Any,
    dataset_meta_features: Union[np.ndarray, List[float], pd.Series],
    n_candidates: int = 200,
    n_select: int = 10,
    beta: float = 0.1,
    seed: int = 42,
    reference_uncertainties: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """End-to-end deterministic pipeline generating confidence-aware hyperparameter recommendations.

    Pipeline:
        dataset meta-features (22 X-only)
                ↓
        generate candidates (n_candidates)
                ↓
        build 30-dimensional candidate matrix
                ↓
        meta-model predictions & tree dispersion uncertainty
                ↓
        relative confidence heuristic
                ↓
        confidence-aware acquisition scoring (mean + beta * unc)
                ↓
        top-N candidate selection

    Parameters
    ----------
    meta_model : fitted ExtraTreesRegressor or MetaModel
        Trained response-surface meta-learner.
    dataset_meta_features : array-like of shape (22,)
        Target dataset's canonical 22 X-only meta-features.
    n_candidates : int, default=200
        Total candidate configurations to sample and evaluate.
    n_select : int, default=10
        Number of top candidate configurations to recommend.
    beta : float, default=0.1
        Exploration weighting parameter.
    seed : int, default=42
        Sampling reproducibility seed.
    reference_uncertainties : np.ndarray, optional
        Reference uncertainty distribution for relative confidence scaling.

    Returns
    -------
    pd.DataFrame
        Top recommended candidate configurations with predicted metrics and acquisition scores.
    """
    # 1. Generate candidate configurations
    candidate_configs = generate_candidate_configs(n_candidates=n_candidates, seed=seed)

    # 2. Build 30-dimensional candidate feature matrix
    candidate_matrix = build_candidate_matrix(dataset_meta_features, candidate_configs)

    # 3. Score candidates with meta-model and estimate uncertainty
    scored_df = score_candidates(
        meta_model=meta_model,
        candidate_matrix=candidate_matrix,
        candidate_configs=candidate_configs,
        reference_uncertainties=reference_uncertainties,
    )

    # 4. Rank and select top candidates via confidence-aware acquisition
    selected_df = select_top_candidates(
        scored_candidates=scored_df,
        n_select=n_select,
        beta=beta,
    )

    return selected_df
