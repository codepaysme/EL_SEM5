"""Unit tests for Phase 11: Final Statistical and Results Analysis."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.final_analysis import (
    analyze_convergence_curves,
    analyze_uncertainty,
    analyze_warm_start,
    build_final_results_table,
    compute_aggregate_descriptive_statistics,
    compute_early_search_comparison,
    compute_method_outcome_summary,
    compute_per_dataset_comparison,
    compute_statistical_tests,
    generate_final_analysis_json,
    generate_report_figures,
    generate_written_report,
    load_benchmark_artifacts,
    run_full_analysis,
)


class TestFinalAnalysis(unittest.TestCase):
    """Test suite for Phase 11 final statistical analysis module."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic DataFrames and temporary directory for tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.out_path = Path(cls.temp_dir)

        # Synthetic summary DataFrame (4 datasets x 3 methods = 12 rows)
        datasets = ["d1", "d2", "d3", "d4"]
        methods = ["random_search", "optuna_tpe", "metahpo_c"]
        summary_records = []
        trial_records = []

        for ds in datasets:
            for m in methods:
                best_cv = 0.70 + (0.05 if m == "metahpo_c" else 0.02)
                test_f1 = 0.68 + (0.04 if m == "metahpo_c" else 0.01)
                test_acc = 0.75 + (0.03 if m == "metahpo_c" else 0.01)
                hpo_time = 10.0
                tot_time = 12.0

                summary_records.append({
                    "dataset_name": ds,
                    "method": m,
                    "best_cv_macro_f1": best_cv,
                    "best_trial_number": 3,
                    "test_macro_f1": test_f1,
                    "test_accuracy": test_acc,
                    "hpo_runtime_seconds": hpo_time,
                    "total_runtime_seconds": tot_time,
                    "n_trials": 20,
                })

                # 20 trials per method
                score = 0.50
                for t in range(20):
                    score += 0.01
                    trial_records.append({
                        "dataset_name": ds,
                        "method": m,
                        "trial_number": t,
                        "cv_macro_f1": score,
                        "best_so_far_cv_macro_f1": score,
                        "duration_seconds": 0.5,
                    })

        cls.df_summary = pd.DataFrame(summary_records)
        cls.df_trials = pd.DataFrame(trial_records)

        # Synthetic LODO uncertainty DataFrame
        unc_records = []
        for i in range(20):
            unc_records.append({
                "dataset_name": "adult" if i < 10 else "credit-g",
                "trial_number": i % 10,
                "actual_cv_macro_f1": 0.8,
                "predicted_cv_macro_f1": 0.75,
                "prediction_uncertainty": 0.05 + i * 0.005,
                "absolute_error": 0.05 + i * 0.004,
                "relative_confidence": 0.6,
            })
        cls.df_lodo_unc = pd.DataFrame(unc_records)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test directory."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_load_benchmark_artifacts(self):
        """Test loading actual Phase 10 benchmark artifacts."""
        df_trials, df_summary, df_curves, metadata, df_lodo_unc = load_benchmark_artifacts()
        self.assertEqual(len(df_trials), 240)
        self.assertEqual(len(df_summary), 12)
        self.assertEqual(len(df_curves), 240)
        self.assertEqual(metadata["n_trials_per_method"], 20)
        self.assertFalse(df_lodo_unc.empty)

    def test_build_final_results_table(self):
        """Test formatting and columns of final results table."""
        orig_summary = self.df_summary.copy()
        df_table = build_final_results_table(self.df_summary, output_dir=self.out_path)

        # Check immutability
        pd.testing.assert_frame_equal(orig_summary, self.df_summary)

        self.assertEqual(len(df_table), 12)
        required_cols = [
            "dataset_name", "method", "best_cv_macro_f1", "test_macro_f1",
            "test_accuracy", "hpo_runtime_seconds", "total_runtime_seconds", "overhead_seconds"
        ]
        for col in required_cols:
            self.assertIn(col, df_table.columns)

        # Verify files created
        self.assertTrue((self.out_path / "final_results_table.csv").exists())
        self.assertTrue((self.out_path / "final_results_table.md").exists())

    def test_compute_per_dataset_comparison(self):
        """Test delta and percentage differences calculation."""
        df_comp = compute_per_dataset_comparison(self.df_summary, output_dir=self.out_path)
        # 4 datasets x 2 baselines = 8 comparisons
        self.assertEqual(len(df_comp), 8)
        self.assertIn("cv_macro_f1_abs_diff", df_comp.columns)
        self.assertIn("cv_macro_f1_pct_diff", df_comp.columns)
        self.assertIn("total_runtime_pct_overhead", df_comp.columns)

        # Ensure no non-finite values
        self.assertTrue(df_comp["cv_macro_f1_abs_diff"].apply(np.isfinite).all())
        self.assertTrue(df_comp["total_runtime_pct_overhead"].apply(np.isfinite).all())

    def test_compute_early_search_comparison(self):
        """Test checkpoint extraction at trials 1, 5, 10, 20."""
        df_early = compute_early_search_comparison(self.df_trials, output_dir=self.out_path)
        # 4 datasets x 4 checkpoints = 16 rows
        self.assertEqual(len(df_early), 16)
        self.assertEqual(sorted(df_early["trial_checkpoint"].unique().tolist()), [1, 5, 10, 20])
        self.assertIn("diff_vs_random", df_early.columns)
        self.assertIn("diff_vs_optuna", df_early.columns)

    def test_analyze_convergence_curves(self):
        """Test convergence trajectory increments and AUC."""
        df_conv = analyze_convergence_curves(self.df_trials, output_dir=self.out_path)
        self.assertEqual(len(df_conv), 12)
        self.assertIn("convergence_auc", df_conv.columns)
        self.assertIn("improvement_1_to_5", df_conv.columns)
        self.assertTrue((df_conv["convergence_auc"] > 0).all())

    def test_analyze_warm_start(self):
        """Test warm start diagnostics for 4 datasets."""
        df_ws = analyze_warm_start(self.df_trials, output_dir=self.out_path)
        self.assertEqual(len(df_ws), 4)
        self.assertIn("warm_start_best_cv_macro_f1", df_ws.columns)
        self.assertIn("best_found_during_warm_start", df_ws.columns)
        self.assertIn("tpe_gain_over_warm_start", df_ws.columns)

    def test_analyze_uncertainty(self):
        """Test uncertainty proxy analysis."""
        df_unc, unc_summary = analyze_uncertainty(self.df_lodo_unc, output_dir=self.out_path)
        self.assertEqual(len(df_unc), 2)
        self.assertIn("overall_spearman_correlation", unc_summary)
        self.assertIn("low_uncertainty_mae", unc_summary)
        self.assertIn("high_uncertainty_mae", unc_summary)

    def test_compute_method_outcome_summary(self):
        """Test outcome summary generator."""
        df_outcome = compute_method_outcome_summary(self.df_summary, self.df_trials, output_dir=self.out_path)
        self.assertEqual(len(df_outcome), 4)
        self.assertIn("top_method_cv_macro_f1", df_outcome.columns)
        self.assertIn("metahpo_c_best_score_reached_at_trial", df_outcome.columns)

    def test_compute_statistical_tests(self):
        """Test paired statistical tests with n=4."""
        df_tests, stat_dict = compute_statistical_tests(self.df_summary, output_dir=self.out_path)
        self.assertGreater(len(df_tests), 0)
        self.assertIn("metahpo_c_vs_random_search_test_macro_f1", stat_dict)
        test_entry = stat_dict["metahpo_c_vs_random_search_test_macro_f1"]
        self.assertEqual(test_entry["n_datasets"], 4)
        self.assertEqual(len(test_entry["paired_differences"]), 4)

    def test_compute_aggregate_descriptive_statistics(self):
        """Test aggregate descriptive statistics across 3 methods."""
        df_agg = compute_aggregate_descriptive_statistics(self.df_summary, self.df_trials, output_dir=self.out_path)
        self.assertEqual(len(df_agg), 3)
        self.assertIn("mean_test_macro_f1", df_agg.columns)
        self.assertIn("median_test_macro_f1", df_agg.columns)

    def test_generate_report_figures(self):
        """Test that all 6 publication figures are generated."""
        fig_paths = generate_report_figures(self.out_path, self.df_summary, self.df_trials, self.df_lodo_unc)
        self.assertEqual(len(fig_paths), 6)
        for name, path in fig_paths.items():
            self.assertTrue(path.exists(), f"Figure {name} not found at {path}")
            self.assertGreater(path.stat().st_size, 0)

    def test_generate_final_analysis_json_and_report(self):
        """Test JSON serialization and markdown report generation."""
        df_comp = compute_per_dataset_comparison(self.df_summary)
        df_early = compute_early_search_comparison(self.df_trials)
        df_ws = analyze_warm_start(self.df_trials)
        _, unc_summary = analyze_uncertainty(self.df_lodo_unc)
        df_agg = compute_aggregate_descriptive_statistics(self.df_summary, self.df_trials)
        df_tests, stat_dict = compute_statistical_tests(self.df_summary)

        json_path = generate_final_analysis_json(
            self.out_path, self.df_summary, df_comp, df_early, df_ws, unc_summary, df_agg, stat_dict
        )
        self.assertTrue(json_path.exists())

        # Verify JSON is loadable and structured correctly
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("final_results", data)
        self.assertIn("methodological_limitations", data)

        md_path = generate_written_report(
            self.out_path, self.df_summary, df_comp, df_early, df_ws, unc_summary, df_agg, df_tests
        )
        self.assertTrue(md_path.exists())
        self.assertGreater(md_path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
