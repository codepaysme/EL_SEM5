"""Unit and integration tests for Phase 6: Meta-Learning Dataset Construction."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import HISTORICAL_DATASETS, get_project_root
from src.features.meta_features import TARGET_META_FEATURE_NAMES, X_ONLY_META_FEATURE_NAMES
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


class TestMetaDataset(unittest.TestCase):
    """Test suite for meta-learning dataset construction, schemas, and leakage protection."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the meta-learning dataset once for fast test evaluation."""
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

    def test_canonical_feature_definitions_and_counts(self) -> None:
        """1, 2, 3, 4: Verify feature name lists, counts, and target name."""
        self.assertEqual(len(X_ONLY_META_FEATURE_NAMES), 22)
        self.assertEqual(len(HPO_PARAMETER_NAMES), 8)
        self.assertEqual(len(META_INPUT_FEATURE_NAMES), 30)
        self.assertEqual(META_INPUT_FEATURE_NAMES, X_ONLY_META_FEATURE_NAMES + HPO_PARAMETER_NAMES)
        self.assertEqual(get_meta_input_feature_names(), META_INPUT_FEATURE_NAMES)
        self.assertEqual(get_hpo_parameter_names(), HPO_PARAMETER_NAMES)
        self.assertEqual(META_TARGET_NAME, "cv_macro_f1")
        self.assertEqual(get_meta_target_name(), "cv_macro_f1")

    def test_meta_dataset_shape_and_counts(self) -> None:
        """5, 6, 7: Verify 140 examples across 7 datasets with 20 trials each."""
        self.assertIsNotNone(self.meta_dataset, "MetaDataset could not be loaded from historical artifacts.")
        df = self.meta_dataset.full_df
        X = self.meta_dataset.X
        y = self.meta_dataset.y

        # Exactly 140 rows
        self.assertEqual(len(df), 140)
        self.assertEqual(len(X), 140)
        self.assertEqual(len(y), 140)

        # X shape is exactly (140, 30)
        self.assertEqual(X.shape, (140, 30))

        # 7 historical datasets present
        unique_datasets = sorted(df["dataset_name"].unique())
        self.assertEqual(unique_datasets, sorted(HISTORICAL_DATASETS))
        self.assertEqual(len(unique_datasets), 7)

        # Exactly 20 trials per dataset
        counts = df.groupby("dataset_name").size().to_dict()
        for ds, count in counts.items():
            self.assertEqual(count, 20, f"Dataset '{ds}' has {count} rows instead of 20.")

    def test_no_missing_or_non_finite_values(self) -> None:
        """8, 9: Verify all predictive features and target contain zero NaNs and all finite values."""
        self.assertIsNotNone(self.meta_dataset)
        X = self.meta_dataset.X
        y = self.meta_dataset.y

        # Feature matrix completeness and finiteness
        self.assertEqual(int(X.isnull().sum().sum()), 0, "Predictive feature matrix X contains null values.")
        self.assertTrue(np.all(np.isfinite(X.to_numpy())), "Predictive feature matrix X contains non-finite values.")

        # Target vector completeness and finiteness
        self.assertEqual(int(y.isnull().sum()), 0, "Target vector y contains null values.")
        self.assertTrue(np.all(np.isfinite(y.to_numpy())), "Target vector y contains non-finite values.")

        # Target metric bounds check for Macro F1 (must be in [0, 1])
        self.assertTrue((y >= 0.0).all() and (y <= 1.0).all(), "Target CV Macro F1 outside [0, 1] range.")

    def test_leakage_protection_in_predictive_features(self) -> None:
        """10, 11, 12, 13: Verify strict absence of leakage and auxiliary columns in predictive features."""
        predictive_cols = set(META_INPUT_FEATURE_NAMES)

        # 10. No target-dependent meta-features
        for target_mf in TARGET_META_FEATURE_NAMES:
            self.assertNotIn(target_mf, predictive_cols, f"Target meta-feature '{target_mf}' leaked into inputs.")

        # Also forbid descriptive label columns
        self.assertNotIn("n_classes_descriptive", predictive_cols)
        self.assertNotIn("n_trials", predictive_cols)

        # 11. No dataset_name in predictive inputs
        self.assertNotIn("dataset_name", predictive_cols)

        # 12. No trial_number in predictive inputs
        self.assertNotIn("trial_number", predictive_cols)

        # 13. No cv_macro_f1 in predictive inputs
        self.assertNotIn("cv_macro_f1", predictive_cols)
        self.assertNotIn("duration_seconds", predictive_cols)
        self.assertNotIn("state", predictive_cols)

    def test_metadata_json_and_save_artifacts(self) -> None:
        """14: Verify serialization and consistency of metadata JSON and CSV export."""
        self.assertIsNotNone(self.meta_dataset)
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path, json_path, feat_path = self.meta_dataset.save(tmp_dir)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(feat_path.exists())

            # Read back saved metadata
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["total_examples"], 140)
            self.assertEqual(metadata["total_predictive_features"], 30)
            self.assertEqual(metadata["meta_feature_count"], 22)
            self.assertEqual(metadata["hyperparameter_count"], 8)
            self.assertEqual(metadata["target_column"], "cv_macro_f1")
            self.assertEqual(len(metadata["historical_datasets"]), 7)
            self.assertEqual(len(metadata["predictive_feature_names"]), 30)

            # Read back saved feature info
            with open(feat_path, "r", encoding="utf-8") as f:
                feat_info = json.load(f)
            self.assertEqual(feat_info["predictive_feature_names"], META_INPUT_FEATURE_NAMES)

    def test_validation_rejects_corrupted_inputs(self) -> None:
        """15: Verify validate_meta_dataset catches missing rows, missing columns, NaNs, and wrong shapes."""
        self.assertIsNotNone(self.meta_dataset)
        df_valid = self.meta_dataset.full_df.copy()

        # Missing row
        df_invalid_rows = df_valid.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            validate_meta_dataset(df_invalid_rows)

        # Missing meta-feature column
        df_missing_col = df_valid.drop(columns=["n_samples"]).copy()
        with self.assertRaises(ValueError):
            validate_meta_dataset(df_missing_col)

        # Missing hyperparameter column
        df_missing_hp = df_valid.drop(columns=["learning_rate"]).copy()
        with self.assertRaises(ValueError):
            validate_meta_dataset(df_missing_hp)

        # NaN in meta-feature
        df_nan_mf = df_valid.copy()
        df_nan_mf.loc[0, "mean_of_feature_means"] = np.nan
        with self.assertRaises(ValueError):
            validate_meta_dataset(df_nan_mf)

        # Infinite in hyperparameter
        df_inf_hp = df_valid.copy()
        df_inf_hp.loc[0, "reg_lambda"] = np.inf
        with self.assertRaises(ValueError):
            validate_meta_dataset(df_inf_hp)


if __name__ == "__main__":
    unittest.main()
