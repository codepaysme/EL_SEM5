"""Unit tests for Phase 9: Confidence-Aware Hyperparameter Candidate Selection."""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.confidence.acquisition import (
    HPO_SEARCH_SPACE,
    build_candidate_matrix,
    compute_acquisition_score,
    generate_candidate_configs,
    recommend_hyperparameters,
    score_candidates,
    select_top_candidates,
)
from src.features.meta_features import X_ONLY_META_FEATURE_NAMES
from src.meta_learning.meta_dataset import HPO_PARAMETER_NAMES, META_INPUT_FEATURE_NAMES


class TestAcquisition(unittest.TestCase):
    """Test suite for candidate sampling, matrix construction, acquisition, and selection."""

    def setUp(self) -> None:
        """Create reproducible synthetic inputs and small meta-model."""
        self.rng = np.random.default_rng(42)
        self.synthetic_m = self.rng.standard_normal(22)

        # Train a small synthetic ExtraTrees model on 30 features
        X_train = pd.DataFrame(
            self.rng.standard_normal((30, len(META_INPUT_FEATURE_NAMES))),
            columns=META_INPUT_FEATURE_NAMES,
        )
        y_train = pd.Series(self.rng.uniform(0.6, 0.95, size=30))
        self.meta_model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        self.meta_model.fit(X_train, y_train)

    def test_candidate_generation_determinism_and_bounds(self) -> None:
        """1. Candidate generation: shape, bounds, keys, and determinism."""
        configs1 = generate_candidate_configs(n_candidates=50, seed=42)
        configs2 = generate_candidate_configs(n_candidates=50, seed=42)

        self.assertEqual(len(configs1), 50)
        self.assertEqual(configs1, configs2)

        for cfg in configs1:
            self.assertEqual(set(cfg.keys()), set(HPO_PARAMETER_NAMES))
            for p, (low, high, ptype) in HPO_SEARCH_SPACE.items():
                val = cfg[p]
                if ptype == "int":
                    self.assertIsInstance(val, int)
                    self.assertGreaterEqual(val, int(low))
                    self.assertLessEqual(val, int(high))
                else:
                    self.assertIsInstance(val, float)
                    self.assertGreaterEqual(val, float(low) - 1e-9)
                    self.assertLessEqual(val, float(high) + 1e-9)

        with self.assertRaises(ValueError):
            generate_candidate_configs(n_candidates=0)

    def test_candidate_matrix_construction(self) -> None:
        """2. Matrix construction: shape, repetition of meta-features, order, validation."""
        configs = generate_candidate_configs(n_candidates=20, seed=42)
        mat = build_candidate_matrix(self.synthetic_m, configs)

        self.assertEqual(mat.shape, (20, 30))

        # First 22 columns must equal synthetic_m across all rows
        for i in range(20):
            self.assertTrue(np.allclose(mat[i, :22], self.synthetic_m))
            # Hyperparameter columns match cfg values
            for j, p in enumerate(HPO_PARAMETER_NAMES):
                self.assertAlmostEqual(mat[i, 22 + j], configs[i][p])

        # Wrong length meta-features
        with self.assertRaises(ValueError):
            build_candidate_matrix(np.zeros(21), configs)

        # Non-finite meta-features
        nan_m = self.synthetic_m.copy()
        nan_m[0] = np.nan
        with self.assertRaises(ValueError):
            build_candidate_matrix(nan_m, configs)

        # Empty configs
        with self.assertRaises(ValueError):
            build_candidate_matrix(self.synthetic_m, [])

    def test_compute_acquisition_score(self) -> None:
        """3. Acquisition scoring: beta=0, scaling with uncertainty, and validation."""
        pred = np.array([0.80, 0.85, 0.70])
        unc = np.array([0.05, 0.10, 0.20])

        # beta = 0 -> acquisition equals prediction
        acq_zero = compute_acquisition_score(pred, unc, beta=0.0)
        self.assertTrue(np.allclose(acq_zero, pred))

        # beta = 0.5 -> acquisition = pred + 0.5 * unc
        acq_half = compute_acquisition_score(pred, unc, beta=0.5)
        expected = pred + 0.5 * unc
        self.assertTrue(np.allclose(acq_half, expected))

        # Larger uncertainty increases acquisition for fixed prediction
        pred_fixed = np.array([0.80, 0.80])
        unc_diff = np.array([0.02, 0.08])
        acq_diff = compute_acquisition_score(pred_fixed, unc_diff, beta=0.1)
        self.assertGreater(acq_diff[1], acq_diff[0])

        # Invalid beta < 0
        with self.assertRaises(ValueError):
            compute_acquisition_score(pred, unc, beta=-0.1)

        # Shape mismatch
        with self.assertRaises(ValueError):
            compute_acquisition_score(pred, np.array([0.05, 0.10]), beta=0.1)

    def test_select_top_candidates_ranking_and_ties(self) -> None:
        """4. Selection: sorting descending, candidate counts, tie-breaking."""
        # Create a mock scored DataFrame
        df_mock = pd.DataFrame({
            "candidate_id": [0, 1, 2, 3],
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.1, 0.05, 0.02, 0.01],
            "subsample": [0.8, 0.8, 0.8, 0.8],
            "colsample_bytree": [0.8, 0.8, 0.8, 0.8],
            "min_child_weight": [1, 2, 3, 4],
            "gamma": [0.0, 0.1, 0.2, 0.3],
            "reg_lambda": [1.0, 1.0, 1.0, 1.0],
            "predicted_score": [0.75, 0.85, 0.80, 0.85],
            "prediction_uncertainty": [0.05, 0.05, 0.10, 0.05],
            "relative_confidence": [0.9, 0.9, 0.8, 0.9],
        })

        # With beta=0.0: candidate 1 and 3 tie with 0.85. Tie-break sorts candidate 1 before 3.
        top2 = select_top_candidates(df_mock, n_select=2, beta=0.0)
        self.assertEqual(len(top2), 2)
        self.assertEqual(top2.iloc[0]["candidate_id"], 1)
        self.assertEqual(top2.iloc[1]["candidate_id"], 3)

        # With beta=1.0:
        # candidate 0: 0.75 + 0.05 = 0.80
        # candidate 1: 0.85 + 0.05 = 0.90
        # candidate 2: 0.80 + 0.10 = 0.90
        # candidate 3: 0.85 + 0.05 = 0.90
        # Candidates 1, 2, 3 tie with 0.90 -> sorted by candidate_id ascending: 1, 2, 3
        top3 = select_top_candidates(df_mock, n_select=3, beta=1.0)
        self.assertEqual(top3["candidate_id"].tolist(), [1, 2, 3])

    def test_recommend_hyperparameters_end_to_end(self) -> None:
        """5. End-to-end recommendation pipeline determinism and column integrity."""
        rec1 = recommend_hyperparameters(
            meta_model=self.meta_model,
            dataset_meta_features=self.synthetic_m,
            n_candidates=50,
            n_select=5,
            beta=0.1,
            seed=42,
        )
        rec2 = recommend_hyperparameters(
            meta_model=self.meta_model,
            dataset_meta_features=self.synthetic_m,
            n_candidates=50,
            n_select=5,
            beta=0.1,
            seed=42,
        )

        self.assertEqual(len(rec1), 5)
        pd.testing.assert_frame_equal(rec1, rec2)

        expected_cols = [
            "candidate_id",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_lambda",
            "predicted_score",
            "prediction_uncertainty",
            "relative_confidence",
            "acquisition_score",
        ]
        self.assertEqual(list(rec1.columns), expected_cols)

        # Check ranges & finiteness
        self.assertTrue(np.all(np.isfinite(rec1["predicted_score"])))
        self.assertTrue(np.all(rec1["prediction_uncertainty"] >= 0.0))
        self.assertTrue(np.all((rec1["relative_confidence"] > 0) & (rec1["relative_confidence"] <= 1.0)))
        self.assertTrue(np.all(np.isfinite(rec1["acquisition_score"])))

        # Sorted descending by acquisition_score
        acq_scores = rec1["acquisition_score"].tolist()
        self.assertEqual(acq_scores, sorted(acq_scores, reverse=True))

    def test_recommend_hyperparameters_with_reference_uncertainties(self) -> None:
        """6. Verify recommendation scaling with supplied reference uncertainties."""
        ref_u = np.array([0.01, 0.02, 0.05, 0.10])
        rec = recommend_hyperparameters(
            meta_model=self.meta_model,
            dataset_meta_features=self.synthetic_m,
            n_candidates=30,
            n_select=3,
            beta=0.2,
            seed=123,
            reference_uncertainties=ref_u,
        )
        self.assertEqual(len(rec), 3)
        self.assertTrue(np.all((rec["relative_confidence"] > 0) & (rec["relative_confidence"] <= 1.0)))

    def test_candidate_matrix_missing_param_raises(self) -> None:
        """7. Verify build_candidate_matrix raises ValueError on incomplete candidate dictionary."""
        bad_configs = [{"n_estimators": 100}]  # Missing other 7 parameters
        with self.assertRaises(ValueError):
            build_candidate_matrix(self.synthetic_m, bad_configs)


if __name__ == "__main__":
    unittest.main()
