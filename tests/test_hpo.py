"""Unit tests for Optuna + XGBoost hyperparameter optimization pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from src.hpo.optimizer import (
    HPOptimizer,
    HPOResult,
    HPOTrialRecord,
    evaluate_fold,
    run_hpo,
    suggest_xgboost_params,
)

EXPECTED_PARAM_NAMES = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_lambda",
}


class TestHPOptimizer(unittest.TestCase):
    """Fast unit tests for HPO search space, execution, reproducibility, and persistence."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.X_synth = pd.DataFrame(
            {
                "feat1": np.random.randn(60),
                "feat2": np.random.uniform(1, 5, size=60),
                "cat1": ["A" if i % 2 == 0 else "B" for i in range(60)],
            }
        )
        self.y_synth = pd.Series([0] * 36 + [1] * 24, name="target")

    def test_suggest_xgboost_params_bounds(self) -> None:
        """Verify that parameter sampler suggests all expected keys within specified bounds."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_xgboost_params(trial)

        self.assertEqual(set(params.keys()), EXPECTED_PARAM_NAMES)
        self.assertTrue(50 <= params["n_estimators"] <= 300)
        self.assertTrue(2 <= params["max_depth"] <= 8)
        self.assertTrue(0.01 <= params["learning_rate"] <= 0.3)
        self.assertTrue(0.6 <= params["subsample"] <= 1.0)
        self.assertTrue(0.6 <= params["colsample_bytree"] <= 1.0)
        self.assertTrue(1 <= params["min_child_weight"] <= 10)
        self.assertTrue(0.0 <= params["gamma"] <= 2.0)
        self.assertTrue(0.1 <= params["reg_lambda"] <= 10.0)

    def test_evaluate_fold_execution(self) -> None:
        """Verify single fold evaluation with leak-free preprocessing returns valid macro F1."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_lambda": 1.0,
        }
        score = evaluate_fold(
            X_train_fold=self.X_synth.iloc[:40],
            y_train_fold=self.y_synth.iloc[:40].to_numpy(),
            X_val_fold=self.X_synth.iloc[40:],
            y_val_fold=self.y_synth.iloc[40:].to_numpy(),
            params=params,
            random_state=42,
        )
        self.assertTrue(0.0 <= score <= 1.0)

    def test_tiny_hpo_smoke(self) -> None:
        """Run a 2-trial fast HPO run on synthetic data and check result attributes."""
        result = run_hpo(
            X=self.X_synth,
            y=self.y_synth,
            dataset_name="synth_test",
            n_trials=2,
            random_state=42,
            n_splits=3,
        )

        self.assertIsInstance(result, HPOResult)
        self.assertEqual(result.dataset_name, "synth_test")
        self.assertEqual(result.n_trials, 2)
        self.assertEqual(len(result.trials), 2)
        self.assertTrue(0.0 <= result.best_score <= 1.0)
        self.assertEqual(set(result.best_params.keys()), EXPECTED_PARAM_NAMES)

        for record in result.trials:
            self.assertIsInstance(record, HPOTrialRecord)
            self.assertEqual(record.state, "COMPLETE")
            self.assertTrue(0.0 <= record.score <= 1.0)
            self.assertGreater(record.duration_seconds, 0.0)

    def test_hpo_reproducibility(self) -> None:
        """Verify that identical random seed produces identical best hyperparameters."""
        res1 = run_hpo(
            X=self.X_synth,
            y=self.y_synth,
            dataset_name="repro_test",
            n_trials=2,
            random_state=42,
            n_splits=3,
        )
        res2 = run_hpo(
            X=self.X_synth,
            y=self.y_synth,
            dataset_name="repro_test",
            n_trials=2,
            random_state=42,
            n_splits=3,
        )

        self.assertAlmostEqual(res1.best_score, res2.best_score, places=5)
        self.assertEqual(res1.best_params, res2.best_params)

    def test_trial_export_and_persistence(self) -> None:
        """Verify DataFrame conversion and CSV/JSON disk export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hpo(
                X=self.X_synth,
                y=self.y_synth,
                dataset_name="export_test",
                n_trials=2,
                random_state=42,
                output_dir=tmp_dir,
            )

            # Check DataFrame
            df = result.to_dataframe()
            self.assertEqual(len(df), 2)
            self.assertIn("cv_macro_f1", df.columns)
            self.assertIn("trial_number", df.columns)
            for param in EXPECTED_PARAM_NAMES:
                self.assertIn(param, df.columns)

            # Check CSV file on disk
            csv_path = Path(tmp_dir) / "export_test_hpo_trials.csv"
            self.assertTrue(csv_path.is_file())
            loaded_df = pd.read_csv(csv_path)
            self.assertEqual(len(loaded_df), 2)

            # Check JSON file on disk
            json_path = Path(tmp_dir) / "export_test_hpo_best.json"
            self.assertTrue(json_path.is_file())
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["dataset"], "export_test")
            self.assertEqual(data["n_trials"], 2)
            self.assertIn("best_cv_macro_f1", data)
            self.assertEqual(set(data["best_params"].keys()), EXPECTED_PARAM_NAMES)

    def test_hpo_optimizer_class_wrapper(self) -> None:
        """Verify HPOptimizer class executes and returns HPOResult."""
        optimizer = HPOptimizer(n_trials=2, random_state=42, n_splits=3)
        result = optimizer.optimize(self.X_synth, self.y_synth, dataset_name="class_test")
        self.assertIsInstance(result, HPOResult)
        self.assertEqual(result.n_trials, 2)


if __name__ == "__main__":
    unittest.main()
