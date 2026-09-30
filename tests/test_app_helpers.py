"""Unit tests for Streamlit application helper functions (Phase 13)."""

import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.confidence.acquisition import HPO_SEARCH_SPACE, build_candidate_matrix, generate_candidate_configs
from src.features.meta_features import X_ONLY_META_FEATURE_NAMES
from src.ui.app_helpers import (
    DEMO_DATASETS,
    META_FEATURE_GROUPS,
    extract_x_only_meta_features_table,
    generate_metahpo_recommendations,
    get_dataset_summary_stats,
    get_grouped_meta_features,
    load_demo_dataset,
    load_meta_model,
    load_validated_benchmark_results,
    parse_uploaded_csv,
)


class TestAppHelpers(unittest.TestCase):
    """Test suite for non-UI Streamlit application helper functions."""

    def test_load_demo_datasets(self):
        """Test loading built-in demo datasets (WDBC, Iris, Wine, Breast Cancer)."""
        for ds_name in DEMO_DATASETS:
            X_df, y_series, title = load_demo_dataset(ds_name)
            self.assertIsInstance(X_df, pd.DataFrame)
            self.assertGreater(len(X_df), 10)
            self.assertGreater(X_df.shape[1], 2)
            self.assertIsInstance(title, str)

        # Invalid demo dataset
        with self.assertRaises(ValueError):
            load_demo_dataset("NonExistentDataset")

    def test_parse_uploaded_csv(self):
        """Test CSV parsing with and without target column, and error handling."""
        # Valid CSV with target
        csv_data = "feat1,feat2,feat3,target\n1,2,3,0\n4,5,6,1\n7,8,9,0\n10,11,12,1\n13,14,15,0\n"
        buffer = io.StringIO(csv_data)
        X_df, y = parse_uploaded_csv(buffer, target_column="target")
        self.assertEqual(X_df.shape, (5, 3))
        self.assertNotIn("target", X_df.columns)
        self.assertEqual(len(y), 5)

        # Valid CSV without target
        buffer = io.StringIO(csv_data)
        X_df_no_target, y_none = parse_uploaded_csv(buffer, target_column=None)
        self.assertEqual(X_df_no_target.shape, (5, 4))
        self.assertIsNone(y_none)

        # Empty CSV
        with self.assertRaises(ValueError):
            parse_uploaded_csv(io.StringIO(""))

        # Too few rows (<5)
        with self.assertRaises(ValueError):
            parse_uploaded_csv(io.StringIO("a,b\n1,2\n3,4\n"))

    def test_get_dataset_summary_stats(self):
        """Test calculation of summary descriptive statistics."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat1": ["a", "b", "c", "d", "e"],
        })
        stats = get_dataset_summary_stats(df)
        self.assertEqual(stats["n_samples"], 5)
        self.assertEqual(stats["n_features"], 3)
        self.assertEqual(stats["numeric_feature_count"], 2)
        self.assertEqual(stats["categorical_feature_count"], 1)
        self.assertAlmostEqual(stats["missing_value_ratio"], 1 / 15)

    def test_extract_x_only_meta_features_table(self):
        """Test that meta-feature extraction uses y=None and yields 22 descriptors."""
        df = pd.DataFrame(np.random.randn(20, 6), columns=[f"col_{i}" for i in range(6)])
        df_table, m_array = extract_x_only_meta_features_table(df, dataset_name="synthetic_test")

        self.assertEqual(len(df_table), 22)
        self.assertEqual(m_array.shape, (22,))
        self.assertEqual(m_array.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(m_array)))
        self.assertEqual(df_table["Meta-Feature"].tolist(), X_ONLY_META_FEATURE_NAMES)

    def test_load_meta_model(self):
        """Test meta-model loading and error handling for missing files."""
        model = load_meta_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

        with self.assertRaises(FileNotFoundError):
            load_meta_model(model_path="non_existent_path_to_model.joblib")

    def test_candidate_matrix_and_generation(self):
        """Test candidate matrix shape (200 x 30) and candidate pool sampling."""
        m_vec = np.random.randn(22)
        candidates = generate_candidate_configs(n_candidates=200, seed=42)
        self.assertEqual(len(candidates), 200)

        X_cand = build_candidate_matrix(m_vec, candidates)
        self.assertEqual(X_cand.shape, (200, 30))

    def test_generate_metahpo_recommendations(self):
        """Test full recommendation pipeline producing top 5 ranked configurations."""
        model = load_meta_model()
        m_vec = np.random.randn(22)

        df_rec = generate_metahpo_recommendations(
            meta_model=model,
            dataset_meta_features=m_vec,
            n_candidates=200,
            n_select=5,
            beta=0.1,
            seed=42,
        )

        self.assertEqual(len(df_rec), 5)
        self.assertEqual(df_rec["Rank"].tolist(), [1, 2, 3, 4, 5])

        # Required metric columns
        for col in ["predicted_cv_macro_f1", "prediction_uncertainty", "relative_confidence", "acquisition_score"]:
            self.assertIn(col, df_rec.columns)

        # All 8 XGBoost hyperparameters present
        for param in HPO_SEARCH_SPACE.keys():
            self.assertIn(param, df_rec.columns)

        # Acquisition score monotonic in top-5
        acq_scores = df_rec["acquisition_score"].tolist()
        self.assertTrue(all(acq_scores[i] >= acq_scores[i + 1] for i in range(len(acq_scores) - 1)))

    def test_meta_feature_groups_and_partitioning(self):
        """Test canonical 6-group meta-feature definitions and partitioning."""
        # Check canonical groups count and feature count
        self.assertEqual(len(META_FEATURE_GROUPS), 6)
        all_grouped_feats = [f for group in META_FEATURE_GROUPS.values() for f in group]
        self.assertEqual(len(all_grouped_feats), 22)
        self.assertEqual(set(all_grouped_feats), set(X_ONLY_META_FEATURE_NAMES))

        # Test partition function
        df = pd.DataFrame(np.random.randn(20, 6), columns=[f"col_{i}" for i in range(6)])
        df_table, _ = extract_x_only_meta_features_table(df)
        grouped = get_grouped_meta_features(df_table)

        self.assertEqual(len(grouped), 6)
        for group_name, feat_list in META_FEATURE_GROUPS.items():
            self.assertIn(group_name, grouped)
            self.assertEqual(len(grouped[group_name]), len(feat_list))
            self.assertEqual(grouped[group_name]["Meta-Feature"].tolist(), feat_list)

    def test_load_validated_benchmark_results(self):
        """Test loading Phase 10/11 benchmark results table."""
        df_bench = load_validated_benchmark_results()
        self.assertEqual(len(df_bench), 12)
        self.assertIn("best_cv_macro_f1", df_bench.columns)
        self.assertIn("test_macro_f1", df_bench.columns)


if __name__ == "__main__":
    unittest.main()
