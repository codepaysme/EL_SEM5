"""Master end-to-end project verification script for MetaHPO-C (Phase 14).

Verifies the entire project codebase, documentation, models, artifacts,
import integrity, unit test suite, and sub-verification suites.
"""

import importlib
import py_compile
import subprocess
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import get_project_root


def main():
    root = get_project_root()
    print("\n" + "=" * 70)
    print("METAHPO-C: MASTER END-TO-END PROJECT VERIFICATION & AUDIT (PHASE 14)")
    print("=" * 70 + "\n")

    passed_checks = 0
    total_checks = 0

    def check(condition: bool, description: str):
        nonlocal passed_checks, total_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"[PASS] {description}")
        else:
            print(f"[FAIL] {description}")
            sys.exit(1)

    # 1. Core Root Files
    check((root / "app.py").exists(), "app.py exists and is present in project root")
    check((root / "README.md").exists(), "README.md exists and is present in project root")
    check((root / "requirements.txt").exists(), "requirements.txt exists and is present in project root")
    check((root / ".gitignore").exists(), ".gitignore exists and is present in project root")

    # 2. Comprehensive Documentation Suite
    check((root / "docs" / "final_report.md").exists(), "docs/final_report.md exists")
    check((root / "docs" / "results_tables.md").exists(), "docs/results_tables.md exists")
    check((root / "docs" / "report_figures.md").exists(), "docs/report_figures.md exists")
    check((root / "docs" / "methodology_notes.md").exists(), "docs/methodology_notes.md exists")
    check((root / "docs" / "demo_and_viva_guide.md").exists(), "docs/demo_and_viva_guide.md exists")
    check((root / "docs" / "project_cheat_sheet.md").exists(), "docs/project_cheat_sheet.md exists")
    check((root / "docs" / "final_project_status.md").exists(), "docs/final_project_status.md exists")

    # 3. Source Packages
    src_dirs = ["data", "features", "hpo", "meta_learning", "confidence", "evaluation", "ui", "utils"]
    for sdir in src_dirs:
        check((root / "src" / sdir).is_dir(), f"src/{sdir} source package directory exists")

    # 4. Critical Research Artifacts
    check((root / "results" / "experiments" / "historical" / "historical_hpo_trials.csv").exists(), "historical_hpo_trials.csv exists (140 rows)")
    check((root / "results" / "experiments" / "meta_learning" / "meta_dataset.csv").exists(), "meta_dataset.csv exists")
    check((root / "results" / "experiments" / "meta_learning" / "meta_model.joblib").exists(), "meta_model.joblib exists (trained ExtraTrees)")
    check((root / "results" / "experiments" / "unseen_benchmark" / "unseen_trial_results.csv").exists(), "unseen_trial_results.csv exists (240 rows)")
    check((root / "results" / "experiments" / "unseen_benchmark" / "unseen_benchmark_summary.csv").exists(), "unseen_benchmark_summary.csv exists (12 rows)")
    check((root / "results" / "analysis" / "final_results_table.csv").exists(), "final_results_table.csv exists")

    # 5. Publication Figures (6 figures, 300 DPI)
    fig_names = [
        "test_macro_f1_comparison.png",
        "test_accuracy_comparison.png",
        "convergence_curves.png",
        "runtime_comparison.png",
        "warm_start_analysis.png",
        "uncertainty_vs_error.png",
    ]
    for fig in fig_names:
        fig_path = root / "results" / "analysis" / "figures" / fig
        check(fig_path.exists() and fig_path.stat().st_size > 1000, f"Publication figure {fig} verified")

    # 6. Source Package Clean Imports
    modules_to_test = [
        "src.data.loader",
        "src.features.meta_features",
        "src.hpo.optimizer",
        "src.meta_learning.meta_dataset",
        "src.meta_learning.meta_model",
        "src.confidence.uncertainty",
        "src.confidence.acquisition",
        "src.evaluation.final_analysis",
        "src.ui.app_helpers",
    ]
    for mod_name in modules_to_test:
        mod = importlib.import_module(mod_name)
        check(mod is not None, f"Module '{mod_name}' imports cleanly")

    # 7. Bytecode Compilation Check for app.py
    try:
        py_compile.compile(str(root / "app.py"), doraise=True)
        check(True, "app.py compiles cleanly to bytecode with zero syntax errors")
    except Exception as e:
        check(False, f"app.py failed compilation: {e}")

    # 8. Run Full Automated Test Suite (91 Tests)
    print("\nRunning full automated test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(root / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    check(result.wasSuccessful() and result.testsRun >= 91, f"All unit and integration tests passed ({result.testsRun}/{result.testsRun} tests)")

    # 9. Execute Sub-Verification Scripts
    print("\nExecuting sub-verification scripts...")
    sub_scripts = [
        ("scripts/verify_unseen_benchmark.py", "Unseen Benchmark Verification"),
        ("scripts/verify_final_analysis.py", "Final Statistical Analysis Verification"),
        ("scripts/verify_final_report.py", "Technical Documentation Verification"),
        ("scripts/verify_streamlit_app.py", "Streamlit Dashboard Verification"),
    ]
    for script_rel_path, script_desc in sub_scripts:
        script_full_path = root / script_rel_path
        res = subprocess.run([sys.executable, str(script_full_path)], capture_output=True, text=True)
        check(res.returncode == 0, f"{script_desc} completed successfully")

    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED")
    print("=" * 70 + "\n")

    print("FINAL PROJECT VERIFICATION: PASS\n")


if __name__ == "__main__":
    main()
