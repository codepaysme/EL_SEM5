"""Verification and audit script for Phase 11 statistical and results analysis."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
from src.data.loader import get_project_root
from src.evaluation.final_analysis import run_full_analysis


def main():
    root = get_project_root()
    analysis_dir = root / "results" / "analysis"
    fig_dir = analysis_dir / "figures"

    # Run analysis if any artifact is missing
    required_files = [
        "final_results_table.csv",
        "final_results_table.md",
        "per_dataset_comparison.csv",
        "early_search_comparison.csv",
        "convergence_summary.csv",
        "warm_start_analysis.csv",
        "uncertainty_analysis.csv",
        "uncertainty_analysis.md",
        "method_outcome_summary.csv",
        "statistical_tests.csv",
        "statistical_analysis.md",
        "aggregate_descriptive_statistics.csv",
        "final_analysis.json",
        "final_results_analysis.md",
    ]
    required_figures = [
        "test_macro_f1_comparison.png",
        "test_accuracy_comparison.png",
        "convergence_curves.png",
        "runtime_comparison.png",
        "warm_start_analysis.png",
        "uncertainty_vs_error.png",
    ]

    all_exist = all((analysis_dir / f).exists() for f in required_files) and all((fig_dir / f).exists() for f in required_figures)
    if not all_exist:
        print("Running Phase 11 analysis pipeline to generate all artifacts...")
        run_full_analysis(output_dir=analysis_dir)

    print("\n=======================================================")
    print("PHASE 11: FINAL STATISTICAL & RESULTS ANALYSIS AUDIT")
    print("=======================================================\n")

    # 1. Audit artifact existence
    for f in required_files:
        p = analysis_dir / f
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty file: {f}"
        print(f"[PASS] Artifact exists: {f} ({p.stat().st_size} bytes)")

    for f in required_figures:
        p = fig_dir / f
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty figure: {f}"
        print(f"[PASS] Figure exists: {f} ({p.stat().st_size} bytes)")

    # 2. Audit JSON structure
    with open(analysis_dir / "final_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["final_results"]) == 12, f"Expected 12 final results in JSON, got {len(data['final_results'])}"
    assert len(data["per_dataset_differences"]) == 8, f"Expected 8 comparisons in JSON, got {len(data['per_dataset_differences'])}"
    print("[PASS] final_analysis.json schema and entry counts verified.")

    # 3. Print key tables
    print("\n" + "=" * 80)
    print("TABLE 1: FINAL RESULTS TABLE")
    print("=" * 80)
    df_table = pd.read_csv(analysis_dir / "final_results_table.csv")
    print(df_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 2: PER-DATASET COMPARISONS (METAHPO-C VS BASELINES)")
    print("=" * 80)
    df_comp = pd.read_csv(analysis_dir / "per_dataset_comparison.csv")
    print(df_comp.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 3: AGGREGATE DESCRIPTIVE STATISTICS ACROSS 4 UNSEEN DATASETS")
    print("=" * 80)
    df_agg = pd.read_csv(analysis_dir / "aggregate_descriptive_statistics.csv")
    print(df_agg.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 4: WARM-START DIAGNOSTIC SUMMARY")
    print("=" * 80)
    df_ws = pd.read_csv(analysis_dir / "warm_start_analysis.csv")
    print(df_ws.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 5: PAIRED STATISTICAL HYPOTHESIS TESTS (N=4 UNSEEN DATASETS)")
    print("=" * 80)
    df_tests = pd.read_csv(analysis_dir / "statistical_tests.csv")
    print(df_tests.to_string(index=False))

    print("\n>>> ALL PHASE 11 CHECKS AND AUDITS PASSED SUCCESSFULLY. <<<\n")


if __name__ == "__main__":
    main()
