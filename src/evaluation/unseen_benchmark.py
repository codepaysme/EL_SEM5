"""Benchmark evaluation of MetaHPO-C vs baselines on reserved unseen datasets.

Evaluation Protocol:
-------------------
1. 4 Reserved Unseen Datasets:
   - madelon, diabetes, steel-plates-fault, ilpd
   - Strictly isolated from all meta-training, meta-model fitting, and uncertainty calibration.

2. Equalized Budget:
   - Exactly 20 actual 3-fold CV evaluations per method per dataset.
   - Zero test-set evaluation during HPO.

3. Methods Evaluated:
   - Method A: Random Search (uniform random sampling over search space, seed=42)
   - Method B: Standard Optuna/TPE (TPESampler, seed=42)
   - Method C: MetaHPO-C (5 confidence-aware warm-start trials + 15 Optuna/TPE trials)

4. Final Model Evaluation:
   - Best hyperparameter configuration selected by CV macro F1.
   - Model retrained on full 80% training partition and evaluated ONCE on 20% test partition.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.confidence.acquisition import (
    HPO_SEARCH_SPACE,
    generate_candidate_configs,
    recommend_hyperparameters,
)
from src.data.loader import (
    UNSEEN_DATASETS,
    get_project_root,
    load_dataset,
    preprocess_features,
    split_dataset,
)
from src.features.meta_features import extract_meta_features, meta_features_to_array
from src.hpo.optimizer import evaluate_fold, suggest_xgboost_params
from src.utils.logging_utils import get_logger

logger = get_logger("unseen_benchmark")

# Suppress verbose Optuna logging during benchmark execution
optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_config_cv(
    X_train_df: pd.DataFrame,
    y_train_encoded: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    params: Dict[str, Any],
    random_state: int = 42,
) -> Tuple[float, float]:
    """Evaluate a single hyperparameter configuration across stratified CV folds.

    Parameters
    ----------
    X_train_df : pd.DataFrame
        Training feature matrix (80% split).
    y_train_encoded : np.ndarray
        Integer-encoded training target labels.
    splits : list of (train_idx, val_idx)
        Stratified K-Fold split indices.
    params : dict
        Hyperparameter configuration dictionary.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    tuple of (float, float)
        (mean_cv_macro_f1, duration_seconds)
    """
    t0 = time.perf_counter()
    fold_scores: List[float] = []

    for train_idx, val_idx in splits:
        X_tr, X_va = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
        y_tr, y_va = y_train_encoded[train_idx], y_train_encoded[val_idx]

        score = evaluate_fold(
            X_train_fold=X_tr,
            y_train_fold=y_tr,
            X_val_fold=X_va,
            y_val_fold=y_va,
            params=params,
            random_state=random_state,
        )
        fold_scores.append(score)

    mean_score = float(np.mean(fold_scores))
    duration = float(time.perf_counter() - t0)
    return mean_score, duration


def evaluate_final_test(
    X_train_df: pd.DataFrame,
    y_train_encoded: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test_encoded: np.ndarray,
    best_params: Dict[str, Any],
    random_state: int = 42,
) -> Tuple[float, float]:
    """Train XGBoost on full training split and evaluate ONCE on held-out test split.

    Parameters
    ----------
    X_train_df : pd.DataFrame
        80% training feature matrix.
    y_train_encoded : np.ndarray
        80% training target vector.
    X_test_df : pd.DataFrame
        20% held-out test feature matrix.
    y_test_encoded : np.ndarray
        20% held-out test target vector.
    best_params : dict
        Best hyperparameter configuration selected by CV.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    tuple of (float, float)
        (test_macro_f1, test_accuracy)
    """
    # Fit preprocessing strictly on training data, transform test data
    X_tr_proc, X_te_proc, _ = preprocess_features(X_train_df, X_test_df)

    num_classes = len(np.unique(y_train_encoded))
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"

    clf = XGBClassifier(
        objective=objective,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
        **best_params,
    )

    clf.fit(X_tr_proc, y_train_encoded)
    y_pred = clf.predict(X_te_proc)

    test_macro_f1 = float(f1_score(y_test_encoded, y_pred, average="macro"))
    test_acc = float(accuracy_score(y_test_encoded, y_pred))

    return test_macro_f1, test_acc


def _config_key(cfg: Dict[str, Any]) -> str:
    """Generate canonical string representation of hyperparameter config for duplicate checking."""
    return "_".join(f"{k}:{cfg[k]}" for k in sorted(cfg.keys()))


def run_random_search_benchmark(
    X_train_df: pd.DataFrame,
    y_train_encoded: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test_encoded: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
    n_trials: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run Random Search benchmark on target dataset."""
    logger.info(f"[{dataset_name}] Running Random Search ({n_trials} trials, seed={seed})...")
    t_start = time.perf_counter()

    # Generate initial pool of random candidate configurations
    candidate_pool = generate_candidate_configs(n_candidates=n_trials * 5, seed=seed)
    evaluated_keys = set()
    trial_records: List[Dict[str, Any]] = []

    best_so_far = -1.0
    pool_idx = 0

    while len(trial_records) < n_trials and pool_idx < len(candidate_pool):
        cfg = candidate_pool[pool_idx]
        pool_idx += 1
        key = _config_key(cfg)
        if key in evaluated_keys:
            continue
        evaluated_keys.add(key)

        trial_num = len(trial_records)
        score, dur = evaluate_config_cv(X_train_df, y_train_encoded, splits, cfg, random_state=seed)

        if score > best_so_far:
            best_so_far = score

        rec = {
            "dataset_name": dataset_name,
            "method": "random_search",
            "trial_number": trial_num,
            "cv_macro_f1": score,
            "best_so_far_cv_macro_f1": best_so_far,
            "duration_seconds": dur,
            **cfg,
        }
        trial_records.append(rec)

    total_time = float(time.perf_counter() - t_start)
    df_trials = pd.DataFrame(trial_records)

    # Find best trial
    best_idx = int(df_trials["cv_macro_f1"].idxmax())
    best_row = df_trials.iloc[best_idx]
    best_params = {p: best_row[p] for p in HPO_SEARCH_SPACE.keys()}
    best_trial_num = int(best_row["trial_number"])
    best_cv_f1 = float(best_row["cv_macro_f1"])

    # Final evaluation on held-out test split
    test_f1, test_acc = evaluate_final_test(
        X_train_df, y_train_encoded, X_test_df, y_test_encoded, best_params, random_state=seed
    )

    summary = {
        "dataset_name": dataset_name,
        "method": "random_search",
        "best_cv_macro_f1": best_cv_f1,
        "best_trial_number": best_trial_num,
        "test_macro_f1": test_f1,
        "test_accuracy": test_acc,
        "hpo_runtime_seconds": float(df_trials["duration_seconds"].sum()),
        "total_runtime_seconds": total_time,
        "n_trials": len(df_trials),
    }

    return df_trials, summary


def run_optuna_benchmark(
    X_train_df: pd.DataFrame,
    y_train_encoded: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test_encoded: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
    n_trials: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run standard cold-start Optuna TPE benchmark on target dataset."""
    logger.info(f"[{dataset_name}] Running Standard Optuna TPE ({n_trials} trials, seed={seed})...")
    t_start = time.perf_counter()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    trial_records: List[Dict[str, Any]] = []
    evaluated_keys = set()
    best_so_far = -1.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_so_far
        params = suggest_xgboost_params(trial)

        # Handle duplicate avoidance
        key = _config_key(params)
        if key in evaluated_keys:
            # Re-sample with perturbed parameter
            params["learning_rate"] = float(params["learning_rate"] * 1.0001)

        evaluated_keys.add(key)
        trial_num = len(trial_records)

        score, dur = evaluate_config_cv(X_train_df, y_train_encoded, splits, params, random_state=seed)

        if score > best_so_far:
            best_so_far = score

        rec = {
            "dataset_name": dataset_name,
            "method": "optuna_tpe",
            "trial_number": trial_num,
            "cv_macro_f1": score,
            "best_so_far_cv_macro_f1": best_so_far,
            "duration_seconds": dur,
            **params,
        }
        trial_records.append(rec)
        return score

    study.optimize(objective, n_trials=n_trials)

    total_time = float(time.perf_counter() - t_start)
    df_trials = pd.DataFrame(trial_records)

    # Best trial
    best_idx = int(df_trials["cv_macro_f1"].idxmax())
    best_row = df_trials.iloc[best_idx]
    best_params = {p: best_row[p] for p in HPO_SEARCH_SPACE.keys()}
    best_trial_num = int(best_row["trial_number"])
    best_cv_f1 = float(best_row["cv_macro_f1"])

    test_f1, test_acc = evaluate_final_test(
        X_train_df, y_train_encoded, X_test_df, y_test_encoded, best_params, random_state=seed
    )

    summary = {
        "dataset_name": dataset_name,
        "method": "optuna_tpe",
        "best_cv_macro_f1": best_cv_f1,
        "best_trial_number": best_trial_num,
        "test_macro_f1": test_f1,
        "test_accuracy": test_acc,
        "hpo_runtime_seconds": float(df_trials["duration_seconds"].sum()),
        "total_runtime_seconds": total_time,
        "n_trials": len(df_trials),
    }

    return df_trials, summary


def run_metahpo_c_benchmark(
    X_train_df: pd.DataFrame,
    y_train_encoded: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test_encoded: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
    meta_model: Any,
    n_trials: int = 20,
    n_warm_start: int = 5,
    beta: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """Run MetaHPO-C confidence-aware warm-started benchmark on target dataset."""
    logger.info(f"[{dataset_name}] Running MetaHPO-C (5 warm-start + 15 TPE trials, seed={seed})...")
    t_start = time.perf_counter()

    # 1. Meta-Inference Phase: Extract 22 X-only meta-features and recommend top 5 warm-start configs
    t_meta_start = time.perf_counter()
    mf_obj = extract_meta_features(X_train_df, y=None, dataset_name=dataset_name)
    m_array = meta_features_to_array(mf_obj, include_target_features=False)

    df_recommended = recommend_hyperparameters(
        meta_model=meta_model,
        dataset_meta_features=m_array,
        n_candidates=200,
        n_select=n_warm_start,
        beta=beta,
        seed=seed,
    )

    warm_start_configs: List[Dict[str, Any]] = []
    for _, row in df_recommended.iterrows():
        cfg = {}
        for p in HPO_SEARCH_SPACE.keys():
            ptype = HPO_SEARCH_SPACE[p][2]
            cfg[p] = int(round(row[p])) if ptype == "int" else float(row[p])
        warm_start_configs.append(cfg)

    meta_inference_time = float(time.perf_counter() - t_meta_start)

    # 2. HPO Execution Phase: Enqueue top 5 warm-start configs, then run TPE exploration
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Enqueue the 5 warm-start candidates
    for cfg in warm_start_configs:
        study.enqueue_trial(cfg)

    trial_records: List[Dict[str, Any]] = []
    evaluated_keys = set()
    best_so_far = -1.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_so_far
        params = suggest_xgboost_params(trial)

        key = _config_key(params)
        if key in evaluated_keys:
            params["learning_rate"] = float(params["learning_rate"] * 1.0001)

        evaluated_keys.add(key)
        trial_num = len(trial_records)

        score, dur = evaluate_config_cv(X_train_df, y_train_encoded, splits, params, random_state=seed)

        if score > best_so_far:
            best_so_far = score

        rec = {
            "dataset_name": dataset_name,
            "method": "metahpo_c",
            "trial_number": trial_num,
            "cv_macro_f1": score,
            "best_so_far_cv_macro_f1": best_so_far,
            "duration_seconds": dur,
            **params,
        }
        trial_records.append(rec)
        return score

    study.optimize(objective, n_trials=n_trials)

    total_time = float(time.perf_counter() - t_start)
    df_trials = pd.DataFrame(trial_records)

    # Best trial
    best_idx = int(df_trials["cv_macro_f1"].idxmax())
    best_row = df_trials.iloc[best_idx]
    best_params = {p: best_row[p] for p in HPO_SEARCH_SPACE.keys()}
    best_trial_num = int(best_row["trial_number"])
    best_cv_f1 = float(best_row["cv_macro_f1"])

    test_f1, test_acc = evaluate_final_test(
        X_train_df, y_train_encoded, X_test_df, y_test_encoded, best_params, random_state=seed
    )

    summary = {
        "dataset_name": dataset_name,
        "method": "metahpo_c",
        "best_cv_macro_f1": best_cv_f1,
        "best_trial_number": best_trial_num,
        "test_macro_f1": test_f1,
        "test_accuracy": test_acc,
        "hpo_runtime_seconds": float(df_trials["duration_seconds"].sum()),
        "total_runtime_seconds": total_time,
        "n_trials": len(df_trials),
    }

    return df_trials, summary, warm_start_configs


def run_unseen_benchmark(
    dataset_names: Optional[List[str]] = None,
    n_trials: int = 20,
    seed: int = 42,
    beta: float = 0.1,
    output_dir: Optional[Union[Path, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run full comparative benchmark across reserved unseen datasets.

    Parameters
    ----------
    dataset_names : list of str, optional
        List of unseen datasets to evaluate. Defaults to UNSEEN_DATASETS.
    n_trials : int, default=20
        Number of trials per method per dataset.
    seed : int, default=42
        Reproducibility seed.
    beta : float, default=0.1
        MetaHPO-C acquisition exploration weight.
    output_dir : Path or str, optional
        Directory to save results. Defaults to results/experiments/unseen_benchmark.

    Returns
    -------
    tuple of (DataFrame, DataFrame, DataFrame, dict)
        (trials_df, summary_df, efficiency_curves_df, metadata_dict)
    """
    if dataset_names is None:
        dataset_names = list(UNSEEN_DATASETS)

    root = get_project_root()
    if output_dir is None:
        out_path = root / "results" / "experiments" / "unseen_benchmark"
    else:
        out_path = Path(output_dir).resolve()

    out_path.mkdir(parents=True, exist_ok=True)

    # Load historical meta-model
    model_path = root / "results" / "experiments" / "meta_learning" / "meta_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained meta-model not found at: {model_path}")

    logger.info(f"Loading historical meta-model from: {model_path.name}")
    meta_model = joblib.load(model_path)

    all_trials: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, Any]] = []
    warm_start_records: Dict[str, List[Dict[str, Any]]] = {}

    total_benchmark_start = time.perf_counter()

    for ds_name in dataset_names:
        logger.info(f"\n==========================================")
        logger.info(f"BENCHMARKING UNSEEN DATASET: {ds_name}")
        logger.info(f"==========================================")

        # 1. Load dataset
        loaded = load_dataset(ds_name)
        X, y = loaded.X, loaded.y

        # 2. Reproducible 80/20 Stratified Train/Test split
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=seed)

        # 3. Label encode target
        le = LabelEncoder()
        y_train_enc = le.fit_transform(np.asarray(y_train))
        y_test_enc = le.transform(np.asarray(y_test))

        # 4. Stratified 3-Fold setup inside training split
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = list(skf.split(X_train, y_train_enc))

        # --- Method A: Random Search ---
        trials_rs, sum_rs = run_random_search_benchmark(
            X_train, y_train_enc, X_test, y_test_enc, splits, ds_name, n_trials=n_trials, seed=seed
        )
        all_trials.append(trials_rs)
        all_summaries.append(sum_rs)

        # --- Method B: Standard Optuna/TPE ---
        trials_opt, sum_opt = run_optuna_benchmark(
            X_train, y_train_enc, X_test, y_test_enc, splits, ds_name, n_trials=n_trials, seed=seed
        )
        all_trials.append(trials_opt)
        all_summaries.append(sum_opt)

        # --- Method C: MetaHPO-C ---
        trials_meta, sum_meta, ws_configs = run_metahpo_c_benchmark(
            X_train, y_train_enc, X_test, y_test_enc, splits, ds_name, meta_model=meta_model,
            n_trials=n_trials, n_warm_start=5, beta=beta, seed=seed
        )
        all_trials.append(trials_meta)
        all_summaries.append(sum_meta)
        warm_start_records[ds_name] = ws_configs

    total_benchmark_time = float(time.perf_counter() - total_benchmark_start)

    # Concatenate all results
    df_all_trials = pd.concat(all_trials, ignore_index=True)
    df_all_summaries = pd.DataFrame(all_summaries)

    # Efficiency curves DataFrame
    df_curves = df_all_trials[["dataset_name", "method", "trial_number", "best_so_far_cv_macro_f1"]].copy()

    # Metadata dictionary
    metadata = {
        "benchmark_name": "MetaHPO-C Unseen Dataset Benchmark",
        "random_seed": seed,
        "n_trials_per_method": n_trials,
        "train_test_split": {"train_ratio": 0.8, "test_ratio": 0.2, "stratified": True, "seed": seed},
        "cv_settings": {"n_splits": 3, "metric": "macro_f1", "stratified": True, "seed": seed},
        "methods": ["random_search", "optuna_tpe", "metahpo_c"],
        "warm_start_count": 5,
        "acquisition_beta": beta,
        "candidate_pool_size": 200,
        "datasets_evaluated": dataset_names,
        "total_benchmark_runtime_seconds": total_benchmark_time,
        "meta_model_source": str(model_path.name),
        "warm_start_configurations_evaluated": warm_start_records,
    }

    # Save artifacts
    trials_csv_path = out_path / "unseen_trial_results.csv"
    summary_csv_path = out_path / "unseen_benchmark_summary.csv"
    curves_csv_path = out_path / "unseen_efficiency_curves.csv"
    metadata_json_path = out_path / "unseen_benchmark_metadata.json"

    df_all_trials.to_csv(trials_csv_path, index=False)
    df_all_summaries.to_csv(summary_csv_path, index=False)
    df_curves.to_csv(curves_csv_path, index=False)

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved trial results ({len(df_all_trials)} rows) to: {trials_csv_path.name}")
    logger.info(f"Saved benchmark summary ({len(df_all_summaries)} rows) to: {summary_csv_path.name}")
    logger.info(f"Saved efficiency curves ({len(df_curves)} rows) to: {curves_csv_path.name}")
    logger.info(f"Saved benchmark metadata to: {metadata_json_path.name}")

    return df_all_trials, df_all_summaries, df_curves, metadata
