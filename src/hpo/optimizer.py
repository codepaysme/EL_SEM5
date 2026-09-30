"""Hyperparameter optimization runner using Optuna and XGBoost with leak-free cross-validation."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data.loader import get_project_root, preprocess_features
from src.utils.logging_utils import get_logger

logger = get_logger("hpo_optimizer")

# Optuna verbosity control (reduce excessive per-trial logging while keeping essentials)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HPOTrialRecord:
    """Record of an individual hyperparameter optimization trial.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset.
    trial_number : int
        Optuna trial index.
    state : str
        Trial completion state ('COMPLETE', 'PRUNED', 'FAIL').
    score : float
        Mean cross-validation macro F1 score (higher is better).
    params : dict of str to Any
        Sampled hyperparameter configuration.
    duration_seconds : float
        Execution time for the trial in seconds.
    """

    dataset_name: str
    trial_number: int
    state: str
    score: float
    params: Dict[str, Any]
    duration_seconds: float


@dataclass
class HPOResult:
    """Summary and complete trajectory of an HPO study.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset.
    n_trials : int
        Total number of trials evaluated.
    best_trial_number : int
        Trial number that achieved the highest CV score.
    best_score : float
        Highest CV macro F1 score achieved.
    best_params : dict of str to Any
        Optimal hyperparameter configuration found.
    trials : list of HPOTrialRecord
        List of all recorded trial trajectories.
    study : optuna.Study, optional
        Underlying Optuna study instance.
    """

    dataset_name: str
    n_trials: int
    best_trial_number: int
    best_score: float
    best_params: Dict[str, Any]
    trials: List[HPOTrialRecord]
    study: Optional[optuna.Study] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trial records into a flattened pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with trial metadata, metric scores, and hyperparameter columns.
        """
        rows = []
        for t in self.trials:
            row = {
                "dataset_name": t.dataset_name,
                "trial_number": t.trial_number,
                "state": t.state,
                "cv_macro_f1": t.score,
                "duration_seconds": t.duration_seconds,
            }
            # Flatten hyperparameter values into distinct columns
            for param_name, param_val in t.params.items():
                row[param_name] = param_val
            rows.append(row)
        return pd.DataFrame(rows)

    def save(
        self,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Tuple[Path, Path]:
        """Save the trial trajectory to CSV and best summary to JSON.

        Parameters
        ----------
        output_dir : Path or str, optional
            Destination directory. Defaults to results/experiments relative to project root.

        Returns
        -------
        tuple of Path
            (csv_path, json_path) pointing to the saved files.
        """
        if output_dir is None:
            out_path = get_project_root() / "results" / "experiments"
        else:
            out_path = Path(output_dir).resolve()

        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Save all trials to CSV
        csv_file = out_path / f"{self.dataset_name}_hpo_trials.csv"
        df_trials = self.to_dataframe()
        df_trials.to_csv(csv_file, index=False)

        # 2. Save best configuration summary to JSON
        json_file = out_path / f"{self.dataset_name}_hpo_best.json"
        best_summary = {
            "dataset": self.dataset_name,
            "n_trials": self.n_trials,
            "best_trial_number": self.best_trial_number,
            "best_cv_macro_f1": float(self.best_score),
            "best_params": self.best_params,
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        return csv_file, json_file


def suggest_xgboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for XGBoost within the compact search space.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial object.

    Returns
    -------
    dict of str to Any
        Sampled hyperparameter dictionary.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def evaluate_fold(
    X_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    X_val_fold: pd.DataFrame,
    y_val_fold: np.ndarray,
    params: Dict[str, Any],
    random_state: int = 42,
) -> float:
    """Fit preprocessing and train XGBoost on one CV fold without data leakage.

    Parameters
    ----------
    X_train_fold : pd.DataFrame
        Training feature matrix for this fold.
    y_train_fold : np.ndarray
        Training labels for this fold.
    X_val_fold : pd.DataFrame
        Validation feature matrix for this fold.
    y_val_fold : np.ndarray
        Validation labels for this fold.
    params : dict
        Hyperparameter configuration for XGBoost.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    float
        Validation macro F1 score for this fold.
    """
    # Fit preprocessing ONLY on fold training data
    X_tr_proc, X_va_proc, _ = preprocess_features(X_train_fold, X_val_fold)

    # Determine objective based on number of unique classes
    num_classes = len(np.unique(y_train_fold))
    if num_classes == 2:
        objective = "binary:logistic"
    else:
        objective = "multi:softprob"

    clf = XGBClassifier(
        objective=objective,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
        **params,
    )

    clf.fit(X_tr_proc, y_train_fold)
    y_pred = clf.predict(X_va_proc)

    return float(f1_score(y_val_fold, y_pred, average="macro"))


def run_hpo(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    dataset_name: str = "dataset",
    n_trials: int = 20,
    random_state: int = 42,
    n_splits: int = 3,
    warm_start_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[Path, str]] = None,
) -> HPOResult:
    """Run hyperparameter optimization for XGBoost using Optuna and leak-free CV.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_samples, n_features)
        Training feature matrix. Note: must NOT include held-out test data.
    y : Series or array-like of shape (n_samples,)
        Training target labels.
    dataset_name : str, default="dataset"
        Identifier for the dataset.
    n_trials : int, default=20
        Number of optimization trials.
    random_state : int, default=42
        Reproducibility seed for TPE sampler and model training.
    n_splits : int, default=3
        Number of stratified cross-validation folds.
    warm_start_params : dict, optional
        Optional initial hyperparameter configuration (e.g. from meta-learning).
    output_dir : Path or str, optional
        Directory to automatically save trial CSV and best JSON.

    Returns
    -------
    HPOResult
        Container with best parameters, CV score, and complete trial records.
    """
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    # Encode labels to uniform non-negative integers (0, 1, ...)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.asarray(y))

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_df, y_encoded))

    # Initialize Optuna Study with TPE sampler
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # If warm-start candidate provided, enqueue as initial trial
    if warm_start_params is not None:
        study.enqueue_trial(warm_start_params)

    trial_records: List[HPOTrialRecord] = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = suggest_xgboost_params(trial)

        fold_scores = []
        for train_idx, val_idx in splits:
            X_tr, X_va = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_tr, y_va = y_encoded[train_idx], y_encoded[val_idx]

            fold_score = evaluate_fold(
                X_train_fold=X_tr,
                y_train_fold=y_tr,
                X_val_fold=X_va,
                y_val_fold=y_va,
                params=params,
                random_state=random_state,
            )
            fold_scores.append(fold_score)

        mean_cv_score = float(np.mean(fold_scores))
        duration = time.perf_counter() - t0

        record = HPOTrialRecord(
            dataset_name=dataset_name,
            trial_number=trial.number,
            state="COMPLETE",
            score=mean_cv_score,
            params=params,
            duration_seconds=duration,
        )
        trial_records.append(record)

        return mean_cv_score

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    result = HPOResult(
        dataset_name=dataset_name,
        n_trials=len(study.trials),
        best_trial_number=best_trial.number,
        best_score=float(best_trial.value),
        best_params=dict(best_trial.params),
        trials=trial_records,
        study=study,
    )

    if output_dir is not None:
        result.save(output_dir=output_dir)

    return result


class HPOptimizer:
    """Hyperparameter optimizer encapsulating XGBoost configuration search.

    Parameters
    ----------
    n_trials : int, default=20
        Number of optimization trials.
    random_state : int, default=42
        Reproducibility seed.
    n_splits : int, default=3
        Number of cross-validation folds.
    """

    def __init__(
        self,
        n_trials: int = 20,
        random_state: int = 42,
        n_splits: int = 3,
    ) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_splits = n_splits

    def optimize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        dataset_name: str = "dataset",
        warm_start_params: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> HPOResult:
        """Execute HPO on the training dataset.

        Parameters
        ----------
        X : DataFrame or array-like
            Training feature matrix.
        y : Series or array-like
            Training target labels.
        dataset_name : str, default="dataset"
            Identifier for the dataset.
        warm_start_params : dict, optional
            Initial hyperparameter candidate.
        output_dir : Path or str, optional
            Path to persist trial trajectories.

        Returns
        -------
        HPOResult
            Container with best configuration, score, and all trial records.
        """
        return run_hpo(
            X=X,
            y=y,
            dataset_name=dataset_name,
            n_trials=self.n_trials,
            random_state=self.random_state,
            n_splits=self.n_splits,
            warm_start_params=warm_start_params,
            output_dir=output_dir,
        )
