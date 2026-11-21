"""Fingerprint-guided hyperparameter search space pruning for MetaHPO-C.

Dynamically contracts XGBoost hyperparameter search space bounds based on
dataset geometric properties (22 X-only meta-features) prior to candidate sampling.
Prevents sampling from parameter regions provably prone to overfitting, collinearity
redundancy, or severe divergence on the target data geometry.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.meta_features import X_ONLY_META_FEATURE_NAMES
from src.confidence.acquisition import HPO_SEARCH_SPACE


def compute_space_volume_ratio(
    base_space: Dict[str, Tuple[Union[int, float], Union[int, float], str]],
    pruned_space: Dict[str, Tuple[Union[int, float], Union[int, float], str]],
) -> float:
    """Calculate the fractional reduction in bounded hyperparameter space volume.

    Parameters
    ----------
    base_space : dict
        Original search space bounds.
    pruned_space : dict
        Contracted search space bounds.

    Returns
    -------
    float
        Percentage volume reduction in [0.0, 100.0].
    """
    base_log_vol = 0.0
    pruned_log_vol = 0.0

    for param, (b_low, b_high, ptype) in base_space.items():
        p_low, p_high, _ = pruned_space[param]
        if ptype == "log_float":
            b_diff = max(np.log(float(b_high)) - np.log(float(b_low)), 1e-9)
            p_diff = max(np.log(float(p_high)) - np.log(float(p_low)), 1e-9)
        else:
            b_diff = max(float(b_high) - float(b_low), 1e-9)
            p_diff = max(float(p_high) - float(p_low), 1e-9)

        base_log_vol += np.log(b_diff)
        pruned_log_vol += np.log(p_diff)

    vol_ratio = float(np.exp(pruned_log_vol - base_log_vol))
    reduction_pct = max(0.0, min(100.0, (1.0 - vol_ratio) * 100.0))
    return reduction_pct


def prune_search_space(
    meta_features: Union[Dict[str, float], np.ndarray, List[float]],
    base_search_space: Optional[Dict[str, Tuple[Union[int, float], Union[int, float], str]]] = None,
) -> Tuple[Dict[str, Tuple[Union[int, float], Union[int, float], str]], List[Dict[str, Any]], float]:
    """Prune XGBoost search space boundaries using the dataset's 22 X-only fingerprint.

    Parameters
    ----------
    meta_features : dict or array-like
        22 X-only meta-features for the target dataset.
    base_search_space : dict, optional
        Baseline search space bounds. Defaults to canonical HPO_SEARCH_SPACE.

    Returns
    -------
    pruned_space : dict
        Pruned search space dictionary with same structure as HPO_SEARCH_SPACE.
    triggered_rules : list of dict
        Descriptive records for every pruning constraint triggered.
    volume_reduction_pct : float
        Estimated percentage of search space pruned away.
    """
    if base_search_space is None:
        pruned_space = {k: (v[0], v[1], v[2]) for k, v in HPO_SEARCH_SPACE.items()}
    else:
        pruned_space = {k: (v[0], v[1], v[2]) for k, v in base_search_space.items()}

    # Convert array to dict if needed
    if isinstance(meta_features, (np.ndarray, list, pd.Series)):
        m_arr = np.asarray(meta_features, dtype=np.float64).flatten()
        if len(m_arr) < len(X_ONLY_META_FEATURE_NAMES):
            raise ValueError(f"Expected at least {len(X_ONLY_META_FEATURE_NAMES)} meta-features, got {len(m_arr)}.")
        m_dict = {name: float(m_arr[i]) for i, name in enumerate(X_ONLY_META_FEATURE_NAMES)}
    elif isinstance(meta_features, dict):
        m_dict = {k: float(v) for k, v in meta_features.items() if v is not None}
    else:
        raise TypeError(f"Unsupported meta_features type: {type(meta_features)}")

    triggered_rules: List[Dict[str, Any]] = []

    n_samples = m_dict.get("n_samples", 1000.0)
    n_features = m_dict.get("n_features", 10.0)
    feat_ratio = m_dict.get("feature_to_sample_ratio", n_features / max(n_samples, 1.0))
    mean_corr = m_dict.get("mean_absolute_correlation", 0.3)
    max_corr = m_dict.get("max_absolute_correlation", 0.5)
    missing_ratio = m_dict.get("missing_value_ratio", 0.0)
    coeff_var = m_dict.get("mean_coefficient_of_variation", 0.8)

    # -------------------------------------------------------------------------
    # RULE 1: Sample Size & Overfitting Constraint
    # -------------------------------------------------------------------------
    if n_samples < 600 or feat_ratio > 0.03:
        old_depth = pruned_space["max_depth"]
        pruned_space["max_depth"] = (old_depth[0], min(4, old_depth[1]), "int")

        old_child = pruned_space["min_child_weight"]
        pruned_space["min_child_weight"] = (max(2, old_child[0]), old_child[1], "int")

        old_sub = pruned_space["subsample"]
        pruned_space["subsample"] = (max(0.70, old_sub[0]), old_sub[1], "float")

        triggered_rules.append({
            "name": "Overfitting Defense Rule",
            "trigger": f"Small sample size (N={int(n_samples):,}) or high feature ratio (P/N={feat_ratio:.4f})",
            "action": "Restricted max_depth to [2, 4], min_child_weight >= 2, subsample >= 0.70",
            "rationale": "Prevents deep decision trees from memorizing sparse individual instances.",
        })
    elif n_samples > 2500:
        old_depth = pruned_space["max_depth"]
        pruned_space["max_depth"] = (max(3, old_depth[0]), old_depth[1], "int")
        triggered_rules.append({
            "name": "Capacity Expansion Rule",
            "trigger": f"Large sample size (N={int(n_samples):,})",
            "action": "Pruned max_depth lower bound to [3, 8]",
            "rationale": "Ensures trees possess sufficient expressive capacity on large tabular datasets.",
        })

    # -------------------------------------------------------------------------
    # RULE 2: Collinearity & Redundancy Dampening
    # -------------------------------------------------------------------------
    if mean_corr > 0.40 or max_corr > 0.80:
        old_col = pruned_space["colsample_bytree"]
        pruned_space["colsample_bytree"] = (old_col[0], min(0.75, old_col[1]), "float")

        triggered_rules.append({
            "name": "Collinearity De-correlation Rule",
            "trigger": f"High feature correlation (mean |r|={mean_corr:.2f}, max |r|={max_corr:.2f})",
            "action": "Capped colsample_bytree upper bound to 0.75",
            "rationale": "Forces trees to sample diverse feature subspaces rather than repeating identical splits on correlated columns.",
        })
    elif n_features <= 6:
        old_col = pruned_space["colsample_bytree"]
        pruned_space["colsample_bytree"] = (max(0.75, old_col[0]), old_col[1], "float")

        triggered_rules.append({
            "name": "Low-Dimensional Feature Visibility Rule",
            "trigger": f"Low feature count (P={int(n_features)})",
            "action": "Raised colsample_bytree lower bound to 0.75",
            "rationale": "Prevents individual trees from starving when the total feature pool is very small.",
        })

    # -------------------------------------------------------------------------
    # RULE 3: Noise & Dispersion Regularization
    # -------------------------------------------------------------------------
    if missing_ratio > 0.005 or coeff_var > 1.2:
        old_lr = pruned_space["learning_rate"]
        pruned_space["learning_rate"] = (old_lr[0], min(0.15, old_lr[1]), "log_float")

        old_gamma = pruned_space["gamma"]
        pruned_space["gamma"] = (max(0.2, old_gamma[0]), old_gamma[1], "float")

        old_lambda = pruned_space["reg_lambda"]
        pruned_space["reg_lambda"] = (max(1.0, old_lambda[0]), old_lambda[1], "log_float")

        triggered_rules.append({
            "name": "Noise Stabilization Rule",
            "trigger": f"Elevated feature dispersion (CV={coeff_var:.2f}) or missingness ({missing_ratio:.2%})",
            "action": "Capped learning_rate to 0.15, raised gamma >= 0.20, raised reg_lambda >= 1.0",
            "rationale": "Slows gradient step size and enforces positive split gain to dampen gradient oscillations.",
        })

    # Compute volume reduction
    base_ref = HPO_SEARCH_SPACE if base_search_space is None else base_search_space
    reduction_pct = compute_space_volume_ratio(base_ref, pruned_space)

    return pruned_space, triggered_rules, reduction_pct
