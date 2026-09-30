"""Unit and integration tests for dataset-level meta-feature extraction (X-only and labeled)."""

import unittest
import numpy as np
import pandas as pd
from src.data.loader import load_dataset
from src.features.meta_features import (
    META_FEATURE_NAMES,
    TARGET_META_FEATURE_NAMES,
    X_ONLY_META_FEATURE_NAMES,
    MetaFeatures,
    extract_meta_features,
    get_meta_feature_names,
    get_target_meta_feature_names,
    get_x_only_meta_feature_names,
    meta_features_to_array,
)


class TestMetaFeatures(unittest.TestCase):
    """Unit tests for meta-feature extractor logic, X-only mode, and labeled mode."""

    def test_meta_feature_names_counts_and_subsets(self) -> None:
        x_only_names = get_x_only_meta_feature_names()
        target_names = get_target_meta_feature_names()
        all_names = get_meta_feature_names(include_target=True)
        x_only_queried = get_meta_feature_names(include_target=False)

        self.assertEqual(len(x_only_names), 22)
        self.assertEqual(len(target_names), 4)
        self.assertEqual(len(all_names), 26)
        self.assertEqual(x_only_names, X_ONLY_META_FEATURE_NAMES)
        self.assertEqual(target_names, TARGET_META_FEATURE_NAMES)
        self.assertEqual(all_names, META_FEATURE_NAMES)
        self.assertEqual(x_only_queried, x_only_names)

    def test_empty_dataset_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_meta_features(pd.DataFrame(), y=None)

    def test_sample_mismatch_raises_value_error(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])
        with self.assertRaises(ValueError):
            extract_meta_features(X, y)

    def test_unlabeled_x_only_extraction(self) -> None:
        """Verify extraction succeeds when y is None without inventing target statistics."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feat1": np.random.randn(50),
                "feat2": np.random.uniform(0, 10, size=50),
                "cat1": ["A" if i % 2 == 0 else "B" for i in range(50)],
            }
        )

        mf = extract_meta_features(X, y=None, dataset_name="unlabeled_synth")

        self.assertIsInstance(mf, MetaFeatures)
        self.assertEqual(mf.dataset_name, "unlabeled_synth")
        self.assertFalse(mf.has_target_features)

        # Verify X-only values are non-None and finite
        for name in X_ONLY_META_FEATURE_NAMES:
            val = mf.values[name]
            self.assertIsNotNone(val, f"X-only feature {name} should not be None")
            self.assertIsInstance(val, (float, int))
            self.assertFalse(np.isnan(val), f"X-only feature {name} is NaN")
            self.assertFalse(np.isinf(val), f"X-only feature {name} is Inf")

        # Verify target-dependent features are explicitly None
        for name in TARGET_META_FEATURE_NAMES:
            self.assertIsNone(mf.values[name], f"Target feature {name} must be None when y=None")

        # Array conversion: X-only (default) produces clean (22,) vector
        arr_x_only = meta_features_to_array(mf, include_target_features=False)
        self.assertEqual(arr_x_only.shape, (22,))
        self.assertEqual(arr_x_only.dtype, np.float64)
        self.assertFalse(np.isnan(arr_x_only).any())

        # Requesting 26-feature vector on unlabeled data must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            meta_features_to_array(mf, include_target_features=True)
        self.assertIn("target-dependent features are unavailable", str(ctx.exception))

    def test_labeled_extraction_and_array_conversion(self) -> None:
        """Verify extraction with ground-truth labels computes all 26 features."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feat1": np.random.randn(50),
                "feat2": np.random.randn(50) * 2.0 + 1.0,
                "feat3": np.random.uniform(0, 10, size=50),
            }
        )
        y = pd.Series([0] * 30 + [1] * 20, name="target")

        mf = extract_meta_features(X, y, dataset_name="labeled_synth")

        self.assertTrue(mf.has_target_features)
        self.assertEqual(len(mf.values), 26)

        # Check target features
        self.assertEqual(mf.values["n_classes"], 2.0)
        self.assertEqual(mf.values["majority_class_ratio"], 0.60)
        self.assertEqual(mf.values["minority_class_ratio"], 0.40)
        self.assertGreater(mf.values["class_entropy"], 0.0)

        # Check array conversions
        arr_22 = meta_features_to_array(mf, include_target_features=False)
        arr_26 = meta_features_to_array(mf, include_target_features=True)

        self.assertEqual(arr_22.shape, (22,))
        self.assertEqual(arr_26.shape, (26,))
        self.assertFalse(np.isnan(arr_22).any())
        self.assertFalse(np.isnan(arr_26).any())
        np.testing.assert_array_equal(arr_22, arr_26[:22])

    def test_synthetic_mixed_dataset_with_missing_values(self) -> None:
        """Verify extraction on mixed numeric/categorical dataset with missing values."""
        X = pd.DataFrame(
            {
                "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
                "num2": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0],
                "cat1": ["low", "high", "low", None, "high", "medium"],
                "cat2": ["A", "A", "B", "B", "A", "B"],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])

        mf = extract_meta_features(X, y, dataset_name="synthetic_mixed")

        self.assertEqual(mf.values["n_samples"], 6.0)
        self.assertEqual(mf.values["n_features"], 4.0)
        self.assertEqual(mf.values["numeric_feature_count"], 2.0)
        self.assertEqual(mf.values["categorical_feature_count"], 2.0)
        self.assertAlmostEqual(mf.values["missing_value_ratio"], 3.0 / 24.0)
        self.assertEqual(mf.values["max_categorical_cardinality"], 3.0)

        arr = meta_features_to_array(mf, include_target_features=False)
        self.assertEqual(arr.shape, (22,))
        self.assertFalse(np.isnan(arr).any())

    def test_constant_numeric_feature_handling(self) -> None:
        """Verify constant columns do not produce NaN or crashes."""
        X = pd.DataFrame(
            {
                "const1": [5.0, 5.0, 5.0, 5.0, 5.0],
                "const2": [0.0, 0.0, 0.0, 0.0, 0.0],
                "var1": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        mf = extract_meta_features(X, y=None, dataset_name="constant_test")

        self.assertEqual(mf.values["n_features"], 3.0)
        for name in X_ONLY_META_FEATURE_NAMES:
            val = mf.values[name]
            self.assertFalse(np.isnan(val), f"Feature {name} became NaN on constant columns")
            self.assertFalse(np.isinf(val), f"Feature {name} became Inf on constant columns")

        arr = meta_features_to_array(mf, include_target_features=False)
        self.assertEqual(arr.shape, (22,))
        self.assertFalse(np.isnan(arr).any())

    def test_x_only_reproducibility(self) -> None:
        """Verify deterministic repeated X-only extractions on identical data."""
        np.random.seed(123)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.uniform(1, 5, size=40),
                "c": np.random.choice(["cat", "dog"], size=40),
            }
        )

        mf1 = extract_meta_features(X, y=None, dataset_name="repro_test")
        mf2 = extract_meta_features(X, y=None, dataset_name="repro_test")

        self.assertEqual(mf1.values, mf2.values)
        arr1 = meta_features_to_array(mf1, include_target_features=False)
        arr2 = meta_features_to_array(mf2, include_target_features=False)
        np.testing.assert_array_equal(arr1, arr2)


class TestMetaFeaturesOpenMLSmoke(unittest.TestCase):
    """Integration smoke test extracting meta-features from live/cached WDBC."""

    def test_wdbc_meta_features_unlabeled_and_labeled(self) -> None:
        try:
            dataset = load_dataset("wdbc")
        except Exception as exc:
            self.skipTest(f"Skipping OpenML smoke test due to network/service error: {exc}")

        # 1. Test X-only extraction (unlabeled mode)
        mf_unlabeled = extract_meta_features(dataset.X, y=None, dataset_name=dataset.name)
        self.assertFalse(mf_unlabeled.has_target_features)
        self.assertEqual(mf_unlabeled.dataset_name, "wdbc")

        for name in X_ONLY_META_FEATURE_NAMES:
            val = mf_unlabeled.values[name]
            self.assertIsNotNone(val)
            self.assertFalse(np.isnan(val))

        for name in TARGET_META_FEATURE_NAMES:
            self.assertIsNone(mf_unlabeled.values[name])

        arr_22 = meta_features_to_array(mf_unlabeled, include_target_features=False)
        self.assertEqual(arr_22.shape, (22,))
        self.assertFalse(np.isnan(arr_22).any())

        # 2. Test labeled extraction
        mf_labeled = extract_meta_features(dataset.X, dataset.y, dataset_name=dataset.name)
        self.assertTrue(mf_labeled.has_target_features)
        self.assertEqual(mf_labeled.values["n_classes"], 2.0)

        arr_26 = meta_features_to_array(mf_labeled, include_target_features=True)
        self.assertEqual(arr_26.shape, (26,))
        self.assertFalse(np.isnan(arr_26).any())

        # Verify X-only portion matches exactly between both modes
        np.testing.assert_array_equal(arr_22, arr_26[:22])


if __name__ == "__main__":
    unittest.main()
