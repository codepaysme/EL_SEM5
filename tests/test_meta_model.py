"""Unit and integration tests for Phase 7: Meta-Learning Model Training and LODO Evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.data.loader import HISTORICAL_DATASETS, get_project_root
from src.meta_learning.meta_dataset import (
    META_INPUT_FEATURE_NAMES,
    META_TARGET_NAME,
    build_meta_dataset,
)
from src.meta_learning.meta_model import (
    DEFAULT_EXTRA_TREES_PARAMS,
    MetaModel,
    compute_regression_metrics,
    evaluate_lodo,
    predict_meta_performance,
    train_final_meta_model,
    train_meta_model,
)


class TestMetaModel(unittest.TestCase):
    """Test suite for meta-model training, LODO cross-validation, and metric computation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the historical meta-dataset once for fast test evaluation."""
        cls.root = get_project_root()
        cls.trials_path = cls.root / "results" / "experiments" / "historical" / "historical_hpo_trials.csv"
        cls.summary_path = cls.root / "results" / "experiments" / "historical" / "historical_dataset_summary.csv"

        if cls.trials_path.exists() and cls.summary_path.exists():
            cls.meta_dataset = build_meta_dataset(
                trials_path=cls.trials_path,
                summary_path=cls.summary_path,
            )
        else:
            cls.meta_dataset = None

    def test_model_instantiation_and_class_interface(self) -> None:
        """1: Verify MetaModel class instantiation and default parameters."""
        meta_learner = MetaModel()
        self.assertFalse(meta_learner.is_fitted)
        self.assertEqual(meta_learner.feature_names, META_INPUT_FEATURE_NAMES)
        self.assertEqual(meta_learner.target_name, META_TARGET_NAME)
        self.assertEqual(meta_learner.model_params["n_estimators"], 300)
        self.assertEqual(meta_learner.model_params["random_state"], 42)

    def test_train_meta_model_and_prediction_shape(self) -> None:
        """2, 3: Verify training function returns fitted model and predictions have correct shape."""
        # Use synthetic fast data
        rng = np.random.RandomState(42)
        X_syn = pd.DataFrame(rng.randn(20, 30), columns=META_INPUT_FEATURE_NAMES)
        y_syn = pd.Series(rng.uniform(0.5, 0.95, size=20))

        model = train_meta_model(X_syn, y_syn, model_params={"n_estimators": 10, "random_state": 42})
        self.assertIsInstance(model, ExtraTreesRegressor)
        self.assertEqual(model.n_features_in_, 30)

        preds = predict_meta_performance(model, X_syn)
        self.assertEqual(preds.shape, (20,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_metric_computation_and_edge_cases(self) -> None:
        """8: Verify metric computation on standard values, constant arrays, and edge cases."""
        y_true = np.array([0.7, 0.8, 0.9])
        y_pred = np.array([0.72, 0.78, 0.91])

        metrics = compute_regression_metrics(y_true, y_pred)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("spearman", metrics)
        self.assertGreaterEqual(metrics["spearman"], 0.9)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

        # Constant prediction edge case (Spearman undefined -> gracefully 0.0)
        metrics_const = compute_regression_metrics(y_true, np.array([0.5, 0.5, 0.5]))
        self.assertEqual(metrics_const["spearman"], 0.0)
        self.assertTrue(np.isfinite(metrics_const["mae"]))

    def test_lodo_evaluation_isolation_and_shapes(self) -> None:
        """4, 5, 6, 7, 9: Verify LODO evaluation partitions cleanly, isolates held-out datasets, and records 140 rows."""
        self.assertIsNotNone(self.meta_dataset)
        df = self.meta_dataset.full_df

        # Run LODO with fast estimator settings for unit test speed
        lodo_res = evaluate_lodo(
            df,
            model_params={"n_estimators": 20, "random_state": 42, "n_jobs": 1},
        )

        # 5. Exactly 7 datasets evaluated
        self.assertEqual(len(lodo_res.per_dataset_metrics), 7)
        self.assertEqual(sorted(lodo_res.per_dataset_metrics.keys()), sorted(HISTORICAL_DATASETS))

        # 6. Exactly 140 prediction rows
        self.assertEqual(len(lodo_res.predictions_df), 140)

        # 7. All predictions and residuals are finite
        pred_cols = ["actual_cv_macro_f1", "predicted_cv_macro_f1", "baseline_predicted_cv_macro_f1", "residual"]
        for col in pred_cols:
            self.assertTrue(
                np.all(np.isfinite(lodo_res.predictions_df[col])),
                f"Non-finite values found in {col}",
            )

        # 9. Baseline predictions exist and are non-null
        self.assertEqual(lodo_res.predictions_df["baseline_predicted_cv_macro_f1"].isnull().sum(), 0)

        # 4. Confirm per-dataset trial count is 20
        ds_counts = lodo_res.predictions_df.groupby("dataset_name").size().to_dict()
        for ds, count in ds_counts.items():
            self.assertEqual(count, 20)

    def test_final_model_training_and_artifacts(self) -> None:
        """10, 11: Verify final model trains on 30 features, saves metadata, and exports feature importances."""
        self.assertIsNotNone(self.meta_dataset)
        df = self.meta_dataset.full_df

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = train_final_meta_model(
                df,
                model_params={"n_estimators": 20, "random_state": 42, "n_jobs": 1},
                save_dir=tmp_dir,
            )

            # 10. Exactly 30 input features
            self.assertEqual(result.model.n_features_in_, 30)
            self.assertEqual(result.training_example_count, 140)

            # 11. Saved feature names match META_INPUT_FEATURE_NAMES exactly
            self.assertEqual(result.feature_names, META_INPUT_FEATURE_NAMES)
            self.assertEqual(len(result.feature_importances), 30)
            self.assertEqual(
                result.feature_importances["feature_name"].tolist(),
                META_INPUT_FEATURE_NAMES,
            )

            # Check saved files on disk
            model_file = Path(tmp_dir) / "meta_model.joblib"
            meta_file = Path(tmp_dir) / "meta_model_metadata.json"
            imp_file = Path(tmp_dir) / "meta_feature_importance.csv"

            self.assertTrue(model_file.exists())
            self.assertTrue(meta_file.exists())
            self.assertTrue(imp_file.exists())

            # Load back model and verify it can predict
            loaded_model = joblib.load(model_file)
            test_x = df[META_INPUT_FEATURE_NAMES].iloc[:5]
            preds = loaded_model.predict(test_x)
            self.assertEqual(len(preds), 5)


if __name__ == "__main__":
    unittest.main()
