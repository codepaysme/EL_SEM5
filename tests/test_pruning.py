"""Unit tests for fingerprint-guided search space pruning."""

import unittest
import numpy as np
import pandas as pd

from src.confidence.acquisition import HPO_SEARCH_SPACE
from src.confidence.pruning import compute_space_volume_ratio, prune_search_space
from src.features.meta_features import X_ONLY_META_FEATURE_NAMES


class TestSearchSpacePruning(unittest.TestCase):
    """Test suite for fingerprint-guided search space contraction."""

    def test_overfitting_defense_small_dataset(self):
        """Test that small sample size triggers max_depth and min_child_weight contraction."""
        meta_dict = {
            "n_samples": 150.0,
            "n_features": 4.0,
            "feature_to_sample_ratio": 4.0 / 150.0,
            "mean_absolute_correlation": 0.2,
            "max_absolute_correlation": 0.4,
            "missing_value_ratio": 0.0,
            "mean_coefficient_of_variation": 0.5,
        }
        pruned_space, rules, red_pct = prune_search_space(meta_dict)
        self.assertGreater(len(rules), 0)
        self.assertLessEqual(pruned_space["max_depth"][1], 4)
        self.assertGreaterEqual(pruned_space["min_child_weight"][0], 2)
        self.assertGreater(red_pct, 0.0)

    def test_collinearity_decorrelation_rule(self):
        """Test that high feature correlation caps colsample_bytree upper bound."""
        meta_dict = {
            "n_samples": 1000.0,
            "n_features": 30.0,
            "feature_to_sample_ratio": 0.03,
            "mean_absolute_correlation": 0.75,
            "max_absolute_correlation": 0.95,
            "missing_value_ratio": 0.0,
            "mean_coefficient_of_variation": 0.6,
        }
        pruned_space, rules, red_pct = prune_search_space(meta_dict)
        self.assertLessEqual(pruned_space["colsample_bytree"][1], 0.75)
        self.assertTrue(any(r["name"] == "Collinearity De-correlation Rule" for r in rules))

    def test_noise_stabilization_rule(self):
        """Test that missingness or high coefficient of variation contracts learning_rate and raises regularization."""
        meta_dict = {
            "n_samples": 1000.0,
            "n_features": 10.0,
            "feature_to_sample_ratio": 0.01,
            "mean_absolute_correlation": 0.2,
            "max_absolute_correlation": 0.3,
            "missing_value_ratio": 0.05,
            "mean_coefficient_of_variation": 2.5,
        }
        pruned_space, rules, red_pct = prune_search_space(meta_dict)
        self.assertLessEqual(pruned_space["learning_rate"][1], 0.15)
        self.assertGreaterEqual(pruned_space["gamma"][0], 0.2)
        self.assertGreaterEqual(pruned_space["reg_lambda"][0], 1.0)

    def test_pruning_with_numpy_array_input(self):
        """Test that prune_search_space functions identically with a 22-D NumPy array."""
        m_arr = np.zeros(22, dtype=np.float64)
        m_arr[0] = 569.0  # n_samples
        m_arr[1] = 30.0   # n_features
        m_arr[2] = 30.0 / 569.0
        m_arr[19] = 0.60  # mean_abs_corr
        m_arr[20] = 0.95  # max_abs_corr

        pruned_space, rules, red_pct = prune_search_space(m_arr)
        self.assertIsInstance(pruned_space, dict)
        self.assertEqual(len(pruned_space), len(HPO_SEARCH_SPACE))
        self.assertGreaterEqual(red_pct, 0.0)
        self.assertLessEqual(red_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
