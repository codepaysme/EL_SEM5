"""Confidence and uncertainty estimation for MetaHPO-C."""

from src.confidence.acquisition import (
    HPO_SEARCH_SPACE,
    build_candidate_matrix,
    compute_acquisition_score,
    generate_candidate_configs,
    recommend_hyperparameters,
    score_candidates,
    select_top_candidates,
)
from src.confidence.estimator import ConfidenceEstimator
from src.confidence.pruning import (
    compute_space_volume_ratio,
    prune_search_space,
)
from src.confidence.uncertainty import (
    LODOUncertaintyResult,
    UncertaintyPrediction,
    estimate_confidence,
    evaluate_lodo_uncertainty,
    predict_with_uncertainty,
)

__all__ = [
    "ConfidenceEstimator",
    "UncertaintyPrediction",
    "LODOUncertaintyResult",
    "predict_with_uncertainty",
    "estimate_confidence",
    "evaluate_lodo_uncertainty",
    "HPO_SEARCH_SPACE",
    "generate_candidate_configs",
    "build_candidate_matrix",
    "score_candidates",
    "compute_acquisition_score",
    "select_top_candidates",
    "recommend_hyperparameters",
    "prune_search_space",
    "compute_space_volume_ratio",
]
