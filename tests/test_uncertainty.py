"""Unit tests for Phase 8: Prediction Uncertainty & Relative Confidence Estimation."""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

from src.confidence.uncertainty import (
    UncertaintyPrediction,
    estimate_confidence,
    predict_with_uncertainty,
)
from src.meta_learning.meta_dataset import META_INPUT_FEATURE_NAMES


class TestUncertainty(unittest.TestCase):
    """Test suite for ensemble dispersion uncertainty and confidence estimation."""

    def setUp(self) -> None:
        """Create synthetic fast models and feature matrices."""
        self.rng = np.random.RandomState(42)
        self.X_syn = pd.DataFrame(
            self.rng.randn(25, len(META_INPUT_FEATURE_NAMES)),
            columns=META_INPUT_FEATURE_NAMES,
        )
        self.y_syn = pd.Series(self.rng.uniform(0.5, 0.95, size=25))

        # Fast fitted ExtraTrees model
        self.model = ExtraTreesRegressor(n_estimators=15, random_state=42)
        self.model.fit(self.X_syn, self.y_syn)

    def test_predict_with_uncertainty_shapes_and_finiteness(self) -> None:
        """1, 2, 3: Verify mean and uncertainty shapes, non-negativity, and finiteness."""
        res = predict_with_uncertainty(self.model, self.X_syn, return_tree_predictions=True)

        self.assertIsInstance(res, UncertaintyPrediction)
        self.assertEqual(res.mean.shape, (25,))
        self.assertEqual(res.std.shape, (25,))
        self.assertIsNotNone(res.tree_predictions)
        self.assertEqual(res.tree_predictions.shape, (15, 25))

        # Finiteness & non-negativity
        self.assertTrue(np.all(np.isfinite(res.mean)))
        self.assertTrue(np.all(np.isfinite(res.std)))
        self.assertTrue(np.all(res.std >= 0.0))

    def test_zero_and_increased_disagreement_properties(self) -> None:
        """5, 6: Verify zero dispersion under perfect agreement and scaling with disagreement."""
        # Mock an ensemble with identical decision trees
        mock_model = ExtraTreesRegressor(n_estimators=5)
        # Train on 1 sample so all trees fit identically
        mock_model.fit(self.X_syn.iloc[:1], self.y_syn.iloc[:1])
        
        # Manually force identical predictions across trees
        for tree in mock_model.estimators_:
            tree.tree_.value[:] = 0.75

        res_zero = predict_with_uncertainty(mock_model, self.X_syn.iloc[:5])
        self.assertTrue(np.allclose(res_zero.std, 0.0, atol=1e-12))
        self.assertTrue(np.allclose(res_zero.mean, 0.75))

        # Test increasing variance
        low_var_preds = np.array([0.74, 0.75, 0.76])
        high_var_preds = np.array([0.40, 0.75, 1.10])
        self.assertGreater(np.std(high_var_preds), np.std(low_var_preds))

    def test_confidence_range_and_monotonicity(self) -> None:
        """4, 7: Verify confidence is bounded in (0, 1] and strictly decreases with uncertainty."""
        uncertainties = np.array([0.0, 0.01, 0.05, 0.10, 0.50, 1.0])

        # Relative confidence
        rel_conf = estimate_confidence(uncertainties, method="relative")
        self.assertEqual(rel_conf.shape, (6,))
        self.assertTrue(np.all(rel_conf > 0.0))
        self.assertTrue(np.all(rel_conf <= 1.0))
        # Zero uncertainty gives highest confidence (1.0)
        self.assertAlmostEqual(rel_conf[0], 1.0)
        # Strictly monotonic decrease
        for i in range(len(rel_conf) - 1):
            self.assertGreater(rel_conf[i], rel_conf[i + 1])

        # Absolute confidence
        abs_conf = estimate_confidence(uncertainties, method="absolute")
        self.assertTrue(np.all(abs_conf > 0.0))
        self.assertTrue(np.all(abs_conf <= 1.0))
        self.assertAlmostEqual(abs_conf[0], 1.0)

    def test_reference_uncertainty_scaling(self) -> None:
        """Verify that supplying reference_uncertainties scales relative confidence consistently."""
        ref_u = np.array([0.02, 0.04, 0.06, 0.08])  # median = 0.05
        test_u = np.array([0.05])

        conf = estimate_confidence(test_u, reference_uncertainties=ref_u, method="relative")
        # When test_u == median(ref_u), normalized_u == 1.0 -> confidence == 1 / (1 + 1) == 0.5
        self.assertAlmostEqual(conf[0], 0.5, places=5)

    def test_batch_ordering_preservation(self) -> None:
        """8: Verify batch prediction ordering matches input feature matrix row-by-row."""
        res_full = predict_with_uncertainty(self.model, self.X_syn)
        res_first = predict_with_uncertainty(self.model, self.X_syn.iloc[[0]])
        res_tenth = predict_with_uncertainty(self.model, self.X_syn.iloc[[9]])

        self.assertAlmostEqual(res_full.mean[0], res_first.mean[0])
        self.assertAlmostEqual(res_full.std[0], res_first.std[0])
        self.assertAlmostEqual(res_full.mean[9], res_tenth.mean[0])
        self.assertAlmostEqual(res_full.std[9], res_tenth.std[0])

    def test_invalid_inputs_raise_appropriate_exceptions(self) -> None:
        """Verify input validation and exception handling."""
        # Non-ensemble model
        single_tree = DecisionTreeRegressor()
        with self.assertRaises(AttributeError):
            predict_with_uncertainty(single_tree, self.X_syn)

        # Negative uncertainty
        with self.assertRaises(ValueError):
            estimate_confidence(np.array([-0.05, 0.10]))


if __name__ == "__main__":
    unittest.main()
