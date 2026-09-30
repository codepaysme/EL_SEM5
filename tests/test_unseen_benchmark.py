"""Unit tests for Phase 10: Reserved unseen dataset benchmark evaluation."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

from src.confidence.acquisition import HPO_SEARCH_SPACE
from src.evaluation.unseen_benchmark import (
    evaluate_config_cv,
    evaluate_final_test,
    run_metahpo_c_benchmark,
    run_optuna_benchmark,
    run_random_search_benchmark,
    run_unseen_benchmark,
)


class TestUnseenBenchmark(unittest.TestCase):
    """Test suite for unseen dataset benchmark runner and components."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic dataset and mock meta-model for fast testing."""
        np.random.seed(42)
        n_samples = 60
        n_features = 10
        X_mat = np.random.randn(n_samples, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        cls.X_df = pd.DataFrame(X_mat, columns=feature_names)
        cls.y = np.array([0] * 30 + [1] * 30)

        # 80/20 split
        cls.X_train = cls.X_df.iloc[:48].reset_index(drop=True)
        cls.y_train = cls.y[:48]
        cls.X_test = cls.X_df.iloc[48:].reset_index(drop=True)
        cls.y_test = cls.y[48:]

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cls.splits = list(skf.split(cls.X_train, cls.y_train))

        cls.sample_params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
        }

        # Train a mock ExtraTreesRegressor on 30 features (22 meta + 8 hyperparams)
        cls.mock_meta_model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        X_mock = np.random.randn(20, 30)
        y_mock = np.random.uniform(0.7, 0.95, size=20)
        cls.mock_meta_model.fit(X_mock, y_mock)

    def test_evaluate_config_cv(self):
        """Test CV evaluation of a single configuration."""
        score, dur = evaluate_config_cv(
            self.X_train, self.y_train, self.splits, self.sample_params, random_state=42
        )
        self.assertIsInstance(score, float)
        self.assertIsInstance(dur, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(dur, 0.0)

    def test_evaluate_final_test(self):
        """Test final evaluation on held-out test split."""
        f1, acc = evaluate_final_test(
            self.X_train, self.y_train, self.X_test, self.y_test, self.sample_params, random_state=42
        )
        self.assertIsInstance(f1, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_run_random_search_benchmark(self):
        """Test Random Search benchmark runner."""
        n_trials = 5
        df_trials, summary = run_random_search_benchmark(
            self.X_train, self.y_train, self.X_test, self.y_test, self.splits,
            dataset_name="synthetic_test", n_trials=n_trials, seed=42
        )

        self.assertEqual(len(df_trials), n_trials)
        self.assertEqual(summary["n_trials"], n_trials)
        self.assertEqual(summary["method"], "random_search")
        self.assertEqual(summary["dataset_name"], "synthetic_test")

        # Check required columns
        required_cols = [
            "dataset_name", "method", "trial_number", "cv_macro_f1", "best_so_far_cv_macro_f1", "duration_seconds"
        ] + list(HPO_SEARCH_SPACE.keys())
        for col in required_cols:
            self.assertIn(col, df_trials.columns)

        # Check monotonic best_so_far
        best_so_far = df_trials["best_so_far_cv_macro_f1"].tolist()
        self.assertTrue(all(best_so_far[i] <= best_so_far[i + 1] for i in range(len(best_so_far) - 1)))
        self.assertAlmostEqual(best_so_far[-1], summary["best_cv_macro_f1"])

    def test_run_optuna_benchmark(self):
        """Test Optuna TPE benchmark runner."""
        n_trials = 5
        df_trials, summary = run_optuna_benchmark(
            self.X_train, self.y_train, self.X_test, self.y_test, self.splits,
            dataset_name="synthetic_test", n_trials=n_trials, seed=42
        )

        self.assertEqual(len(df_trials), n_trials)
        self.assertEqual(summary["n_trials"], n_trials)
        self.assertEqual(summary["method"], "optuna_tpe")

        # Check monotonic best_so_far
        best_so_far = df_trials["best_so_far_cv_macro_f1"].tolist()
        self.assertTrue(all(best_so_far[i] <= best_so_far[i + 1] for i in range(len(best_so_far) - 1)))
        self.assertAlmostEqual(best_so_far[-1], summary["best_cv_macro_f1"])

    def test_run_metahpo_c_benchmark(self):
        """Test MetaHPO-C warm-started benchmark runner."""
        n_trials = 5
        n_warm = 2
        df_trials, summary, ws_configs = run_metahpo_c_benchmark(
            self.X_train, self.y_train, self.X_test, self.y_test, self.splits,
            dataset_name="synthetic_test", meta_model=self.mock_meta_model,
            n_trials=n_trials, n_warm_start=n_warm, beta=0.1, seed=42
        )

        self.assertEqual(len(df_trials), n_trials)
        self.assertEqual(len(ws_configs), n_warm)
        self.assertEqual(summary["n_trials"], n_trials)
        self.assertEqual(summary["method"], "metahpo_c")

        # Check monotonic best_so_far
        best_so_far = df_trials["best_so_far_cv_macro_f1"].tolist()
        self.assertTrue(all(best_so_far[i] <= best_so_far[i + 1] for i in range(len(best_so_far) - 1)))
        self.assertAlmostEqual(best_so_far[-1], summary["best_cv_macro_f1"])

        # Check that the first n_warm trials correspond to the warm-start configurations
        for i in range(n_warm):
            trial_row = df_trials.iloc[i]
            ws_cfg = ws_configs[i]
            for p in HPO_SEARCH_SPACE.keys():
                self.assertAlmostEqual(trial_row[p], ws_cfg[p], places=4)


if __name__ == "__main__":
    unittest.main()
