"""Unit and integration tests for dataset loading, validation, splitting, and preprocessing."""

import unittest
import numpy as np
import pandas as pd
from src.data.loader import (
    LoadedDataset,
    build_feature_preprocessor,
    load_dataset,
    preprocess_features,
    split_dataset,
    validate_dataset,
)


class TestLoaderValidation(unittest.TestCase):
    """Test configuration lookups and validation error conditions."""

    def test_nonexistent_dataset_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            load_dataset("non_existent_dataset_xyz")

    def test_sample_mismatch_raises_value_error(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(X, y, dataset_name="mismatch_test")
        self.assertIn("Sample count mismatch", str(ctx.exception))

    def test_empty_dataset_raises_value_error(self) -> None:
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(X, y, dataset_name="empty_test")
        self.assertIn("zero samples", str(ctx.exception))

    def test_single_class_target_raises_value_error(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([1, 1, 1, 1])
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(X, y, dataset_name="single_class_test")
        self.assertIn("at least 2 distinct target classes", str(ctx.exception))

    def test_all_missing_target_raises_value_error(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([np.nan, np.nan, np.nan, np.nan])
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(X, y, dataset_name="nan_target_test")
        self.assertIn("missing values", str(ctx.exception))


class TestSplitAndPreprocessing(unittest.TestCase):
    """Test splitting logic and leakage-free feature preprocessing."""

    def setUp(self) -> None:
        # Create a synthetic dataset with 100 samples
        np.random.seed(42)
        self.X_df = pd.DataFrame(
            {
                "num1": np.random.randn(100),
                "num2": [np.nan if i % 10 == 0 else float(i) for i in range(100)],
                "cat1": ["A" if i % 2 == 0 else "B" for i in range(100)],
            }
        )
        self.y_series = pd.Series([0 if i < 60 else 1 for i in range(100)], name="label")

    def test_invalid_test_size_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            split_dataset(self.X_df, self.y_series, test_size=1.5)
        with self.assertRaises(ValueError):
            split_dataset(self.X_df, self.y_series, test_size=-0.1)

    def test_rare_class_in_split_raises_value_error(self) -> None:
        y_rare = pd.Series([0] * 50 + [1] * 49 + [2])  # Class '2' has only 1 sample
        with self.assertRaises(ValueError) as ctx:
            split_dataset(self.X_df, y_rare, test_size=0.2)
        self.assertIn("fewer than 2 samples", str(ctx.exception))

    def test_split_sizes_and_stratification(self) -> None:
        X_train, X_test, y_train, y_test = split_dataset(
            self.X_df,
            self.y_series,
            test_size=0.2,
            random_state=42,
        )
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

        # Check stratification proportions (60% class 0, 40% class 1)
        train_ratio = (y_train == 0).mean()
        test_ratio = (y_test == 0).mean()
        self.assertAlmostEqual(train_ratio, 0.60, places=2)
        self.assertAlmostEqual(test_ratio, 0.60, places=2)

    def test_split_reproducibility(self) -> None:
        X_tr1, X_te1, y_tr1, y_te1 = split_dataset(self.X_df, self.y_series, test_size=0.2, random_state=42)
        X_tr2, X_te2, y_tr2, y_te2 = split_dataset(self.X_df, self.y_series, test_size=0.2, random_state=42)
        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_series_equal(y_tr1, y_tr2)

    def test_preprocessing_pipeline_handles_missing_and_categoricals(self) -> None:
        X_train, X_test, _, _ = split_dataset(self.X_df, self.y_series, test_size=0.2, random_state=42)
        X_train_proc, X_test_proc, preprocessor = preprocess_features(X_train, X_test)

        self.assertIsNotNone(X_test_proc)
        self.assertEqual(X_train_proc.shape[0], 80)
        self.assertEqual(X_test_proc.shape[0], 20)
        self.assertEqual(X_train_proc.shape[1], X_test_proc.shape[1])

        # Verify no NaN values remain after preprocessing
        self.assertFalse(np.isnan(X_train_proc).any())
        self.assertFalse(np.isnan(X_test_proc).any())

    def test_preprocessing_handles_unseen_categories_in_test(self) -> None:
        X_train = pd.DataFrame({"cat": ["A", "B", "A", "B"], "num": [1.0, 2.0, 3.0, 4.0]})
        X_test = pd.DataFrame({"cat": ["C", "A"], "num": [5.0, 6.0]})  # 'C' is unseen

        X_train_proc, X_test_proc, _ = preprocess_features(X_train, X_test)
        self.assertEqual(X_train_proc.shape[0], 4)
        self.assertEqual(X_test_proc.shape[0], 2)
        self.assertEqual(X_train_proc.shape[1], X_test_proc.shape[1])
        self.assertFalse(np.isnan(X_test_proc).any())


class TestOpenMLIntegration(unittest.TestCase):
    """Integration smoke test fetching WDBC dataset from OpenML."""

    def test_wdbc_openml_smoke(self) -> None:
        try:
            dataset = load_dataset("wdbc")
        except Exception as exc:
            self.skipTest(f"Skipping OpenML live smoke test due to network/service error: {exc}")

        self.assertIsInstance(dataset, LoadedDataset)
        self.assertEqual(dataset.name, "wdbc")
        self.assertEqual(dataset.task_type, "classification")
        self.assertGreater(dataset.n_samples, 0)
        self.assertGreater(dataset.n_features, 1)
        self.assertEqual(len(dataset.X), dataset.n_samples)
        self.assertEqual(len(dataset.y), dataset.n_samples)
        self.assertEqual(len(dataset.feature_names), dataset.n_features)

        # Verify target has at least 2 distinct classes
        unique_targets = dataset.y.dropna().unique()
        self.assertGreaterEqual(len(unique_targets), 2)

        # Verify splitting works with loaded dataset
        X_train, X_test, y_train, y_test = split_dataset(
            dataset.X,
            dataset.y,
            test_size=0.2,
            random_state=42,
        )
        self.assertEqual(len(X_train) + len(X_test), dataset.n_samples)
        self.assertEqual(len(y_train) + len(y_test), dataset.n_samples)

        # Verify preprocessing works
        X_train_proc, X_test_proc, _ = preprocess_features(X_train, X_test)
        self.assertEqual(X_train_proc.shape[0], len(X_train))
        self.assertEqual(X_test_proc.shape[0], len(X_test))
        self.assertFalse(np.isnan(X_train_proc).any())
        self.assertFalse(np.isnan(X_test_proc).any())


if __name__ == "__main__":
    unittest.main()
