"""Unit tests for Phase 5: Historical experiment runner, schema integrity, and unseen isolation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.run_historical_hpo import (
    run_all_historical_datasets,
    run_historical_dataset,
    save_meta_features_json,
)
from src.data.loader import (
    HISTORICAL_DATASETS,
    UNSEEN_DATASETS,
    get_historical_dataset_names,
    get_unseen_dataset_names,
)
from src.features.meta_features import (
    X_ONLY_META_FEATURE_NAMES,
    extract_meta_features,
    get_x_only_meta_feature_names,
)


class TestHistoricalExperiment(unittest.TestCase):
    """Unit tests for historical HPO workflow and data schemas."""

    def test_historical_and_unseen_separation(self) -> None:
        """Test A: Verify historical and unseen pools are disjoint and contain expected datasets."""
        hist = get_historical_dataset_names()
        unseen = get_unseen_dataset_names()

        self.assertEqual(len(hist), 7)
        self.assertEqual(len(unseen), 4)

        hist_set = set(hist)
        unseen_set = set(unseen)

        # Zero intersection
        self.assertEqual(hist_set & unseen_set, set(), "Historical and unseen pools must be strictly disjoint.")

        expected_hist = {"adult", "bank-marketing", "credit-g", "wdbc", "spambase", "vehicle", "phoneme"}
        expected_unseen = {"madelon", "diabetes", "steel-plates-fault", "ilpd"}

        self.assertEqual(hist_set, expected_hist)
        self.assertEqual(unseen_set, expected_unseen)

    def test_meta_feature_artifact_format(self) -> None:
        """Test B: Verify meta-feature JSON artifact contains exactly 22 X-only features."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
        mf = extract_meta_features(X, y=None, dataset_name="test_ds")

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = save_meta_features_json(mf.values, "test_ds", Path(tmp_dir))
            self.assertTrue(json_path.is_file())

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["dataset_name"], "test_ds")
            self.assertEqual(data["n_meta_features"], 22)
            self.assertEqual(data["feature_names"], get_x_only_meta_feature_names())
            self.assertEqual(len(data["features"]), 22)
            self.assertEqual(set(data["features"].keys()), set(X_ONLY_META_FEATURE_NAMES))

    def test_unseen_dataset_protection(self) -> None:
        """Test E: Verify runner raises ValueError if any unseen dataset is provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for unseen_name in UNSEEN_DATASETS:
                with self.assertRaises(ValueError) as ctx:
                    run_historical_dataset(dataset_name=unseen_name, output_dir=Path(tmp_dir))
                self.assertIn("reserved as a held-out unseen test dataset", str(ctx.exception))

                with self.assertRaises(ValueError) as ctx2:
                    run_all_historical_datasets(dataset_names=[unseen_name], output_dir=Path(tmp_dir))
                self.assertIn("Historical HPO runners must NOT touch unseen test datasets", str(ctx2.exception))

    def test_historical_runner_fast_smoke_and_schemas(self) -> None:
        """Test C & D: Fast smoke test verifying trial and summary CSV schemas."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)

            # Run 1 dataset with 2 trials for speed
            run_all_historical_datasets(
                dataset_names=["wdbc"],
                n_trials=2,
                random_state=42,
                output_dir=out_path,
            )

            # Check individual files
            self.assertTrue((out_path / "wdbc_hpo_trials.csv").is_file())
            self.assertTrue((out_path / "wdbc_hpo_best.json").is_file())
            self.assertTrue((out_path / "wdbc_meta_features.json").is_file())

            # Check aggregate trials CSV (Test C)
            agg_trials_path = out_path / "historical_hpo_trials.csv"
            self.assertTrue(agg_trials_path.is_file())
            df_trials = pd.read_csv(agg_trials_path)
            self.assertEqual(len(df_trials), 2)
            expected_trial_cols = {
                "dataset_name",
                "trial_number",
                "state",
                "cv_macro_f1",
                "duration_seconds",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "gamma",
                "reg_lambda",
            }
            self.assertTrue(expected_trial_cols.issubset(set(df_trials.columns)))

            # Check aggregate summary CSV (Test D)
            agg_summary_path = out_path / "historical_dataset_summary.csv"
            self.assertTrue(agg_summary_path.is_file())
            df_summary = pd.read_csv(agg_summary_path)
            self.assertEqual(len(df_summary), 1)

            # Verify all 22 X-only meta-feature columns exist
            for feat_name in X_ONLY_META_FEATURE_NAMES:
                self.assertIn(feat_name, df_summary.columns)

            self.assertIn("best_cv_macro_f1", df_summary.columns)
            self.assertIn("best_trial_number", df_summary.columns)
            self.assertIn("n_trials", df_summary.columns)
            self.assertIn("n_samples", df_summary.columns)
            self.assertIn("n_features", df_summary.columns)


if __name__ == "__main__":
    unittest.main()
