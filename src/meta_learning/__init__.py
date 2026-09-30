"""Meta-learning models and dataset construction for MetaHPO-C."""

from src.meta_learning.meta_dataset import (
    AUXILIARY_COLUMNS,
    HPO_PARAMETER_NAMES,
    META_INPUT_FEATURE_NAMES,
    META_TARGET_NAME,
    MetaDataset,
    build_meta_dataset,
    get_hpo_parameter_names,
    get_meta_input_feature_names,
    get_meta_target_name,
    validate_meta_dataset,
)
from src.meta_learning.meta_model import (
    DEFAULT_EXTRA_TREES_PARAMS,
    LODOResult,
    MetaModel,
    MetaModelResult,
    compute_regression_metrics,
    evaluate_lodo,
    predict_meta_performance,
    train_final_meta_model,
    train_meta_model,
)

__all__ = [
    "MetaModel",
    "MetaDataset",
    "MetaModelResult",
    "LODOResult",
    "build_meta_dataset",
    "validate_meta_dataset",
    "get_meta_input_feature_names",
    "get_hpo_parameter_names",
    "get_meta_target_name",
    "HPO_PARAMETER_NAMES",
    "META_INPUT_FEATURE_NAMES",
    "META_TARGET_NAME",
    "AUXILIARY_COLUMNS",
    "DEFAULT_EXTRA_TREES_PARAMS",
    "compute_regression_metrics",
    "evaluate_lodo",
    "predict_meta_performance",
    "train_final_meta_model",
    "train_meta_model",
]

