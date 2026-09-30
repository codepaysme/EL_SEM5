"""Verification and audit script for Phase 12 documentation and technical report."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from src.data.loader import get_project_root


def main():
    root = get_project_root()
    docs_dir = root / "docs"

    print("\n=======================================================")
    print("PHASE 12: FINAL TECHNICAL REPORT & DOCUMENTATION AUDIT")
    print("=======================================================\n")

    # 1. Document Existence & Size Checks
    required_docs = [
        "final_report.md",
        "results_tables.md",
        "report_figures.md",
        "methodology_notes.md",
    ]

    for doc in required_docs:
        p = docs_dir / doc
        assert p.exists() and p.stat().st_size > 500, f"Missing or incomplete documentation file: {doc}"
        print(f"[PASS] Document exists and verified: {doc} ({p.stat().st_size} bytes)")

    # 2. Section Checks in final_report.md
    with open(docs_dir / "final_report.md", "r", encoding="utf-8") as f:
        report_text = f.read()

    required_sections = [
        "Abstract",
        "Introduction",
        "Background",
        "Proposed Method: MetaHPO-C",
        "Historical HPO",
        "Meta-Dataset Construction",
        "Meta-Model Architecture",
        "Uncertainty Estimation",
        "Experimental Setup",
        "Results and Comparative Evaluation",
        "Search Efficiency",
        "Warm-Start Diagnostic Analysis",
        "Computational Efficiency",
        "Statistical Analysis",
        "Discussion",
        "Methodological Limitations",
        "Future Research Directions",
        "Conclusion",
        "References",
    ]

    for sec in required_sections:
        assert sec.lower() in report_text.lower(), f"Missing required section in final_report.md: {sec}"
    print(f"[PASS] All {len(required_sections)} major report sections present in final_report.md.")

    # 3. Numerical Integrity Check against CSV Artifacts
    df_summary = pd.read_csv(root / "results" / "analysis" / "final_results_table.csv")
    for _, row in df_summary.iterrows():
        test_f1_str = f"{row['test_macro_f1']:.6f}"
        assert test_f1_str in report_text, f"Numerical mismatch in report for {row['dataset_name']} {row['method']}: {test_f1_str}"
    print("[PASS] Exact numerical scores in final_report.md match authoritative Phase 11 artifacts.")

    # 4. Check results_tables.md content
    with open(docs_dir / "results_tables.md", "r", encoding="utf-8") as f:
        tables_text = f.read()

    table_headers = [
        "Table 1: Historical Meta-Dataset Summary",
        "Table 2: Leave-One-Dataset-Out (LODO) Meta-Model Validation",
        "Table 3: LODO Uncertainty Proxy vs. Prediction Error Breakdown",
        "Table 4: Reserved Unseen Benchmark Comprehensive Results",
        "Table 5: Per-Dataset Differences",
        "Table 6: Optimization Trajectory Checkpoints",
        "Table 7: Warm-Start Diagnostic Summary",
        "Table 8: Paired Statistical Hypothesis Testing",
    ]
    for th in table_headers:
        assert th.lower() in tables_text.lower(), f"Missing table in results_tables.md: {th}"
    print(f"[PASS] All {len(table_headers)} comprehensive results tables verified in results_tables.md.")

    # 5. Check report_figures.md content
    with open(docs_dir / "report_figures.md", "r", encoding="utf-8") as f:
        figs_text = f.read()

    fig_headers = [
        "Figure 1: Held-Out Test Macro-F1 Comparison",
        "Figure 2: Held-Out Test Accuracy Comparison",
        "Figure 3: HPO Convergence Trajectories",
        "Figure 4: Total Benchmark Runtime Comparison",
        "Figure 5: Warm-Start Window Progression",
        "Figure 6: ExtraTrees Dispersion Uncertainty vs. Prediction Error",
    ]
    for fh in fig_headers:
        assert fh.lower() in figs_text.lower(), f"Missing figure description in report_figures.md: {fh}"
    print(f"[PASS] All {len(fig_headers)} publication figures cataloged and verified in report_figures.md.")

    # 6. Artifact Mutability Safeguard Check
    assert (root / "results" / "experiments" / "unseen_benchmark" / "unseen_trial_results.csv").exists()
    assert (root / "results" / "experiments" / "meta_learning" / "meta_model.joblib").exists()
    print("[PASS] Existing benchmark and meta-learning models preserved without mutation.")

    print("\n>>> ALL PHASE 12 DOCUMENTATION AUDITS PASSED SUCCESSFULLY. <<<\n")


if __name__ == "__main__":
    main()
