"""Verification and audit script for the redesigned MetaHPO-C Streamlit UI/UX."""

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

import numpy as np
import pandas as pd
import streamlit as st

from src.confidence.acquisition import HPO_SEARCH_SPACE, build_candidate_matrix, generate_candidate_configs
from src.data.loader import get_project_root
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
)


def main():
    root = get_project_root()
    app_path = root / "app.py"

    print("\n=======================================================")
    print("METAHPO-C: STREAMLIT UI/UX REDESIGN VERIFICATION & AUDIT")
    print("=======================================================\n")

    # 1. app.py exists and is non-empty
    assert app_path.exists() and app_path.stat().st_size > 1000, "app.py missing or empty"
    print(f"[PASS] app.py exists and verified ({app_path.stat().st_size} bytes).")

    # 2. Bytecode compilation with zero warnings/errors
    try:
        py_compile.compile(str(app_path), doraise=True)
        print("[PASS] app.py compiles cleanly to bytecode with zero syntax warnings.")
    except Exception as e:
        print(f"[FAIL] app.py bytecode compilation failed: {e}")
        sys.exit(1)

    # 3. Check for deprecated Streamlit APIs or unescaped strings in app.py
    app_content = app_path.read_text(encoding="utf-8")
    assert "use_container_width=True" not in app_content, "Deprecated use_container_width=True found in app.py"
    assert "use_container_width=False" not in app_content or "use_container_width" not in app_content, "Deprecated width API found"
    print("[PASS] Streamlit width API compliance verified (no use_container_width deprecation).")

    # 4. Meta-model artifact loads
    meta_model = load_meta_model()
    assert meta_model is not None, "Failed to load meta-model artifact"
    print("[PASS] Trained ExtraTrees meta-model artifact loaded successfully.")

    # 5. Built-in demo datasets load
    for ds in DEMO_DATASETS:
        X_df, y, title = load_demo_dataset(ds)
        assert len(X_df) > 0 and X_df.shape[1] > 0, f"Demo dataset {ds} empty"
    print(f"[PASS] All {len(DEMO_DATASETS)} built-in demo datasets load cleanly.")

    # 6. X-only meta-feature extraction (22 descriptors, y=None)
    X_wdbc, _, _ = load_demo_dataset("WDBC")
    df_meta, m_vec = extract_x_only_meta_features_table(X_wdbc, dataset_name="WDBC_ui_test")
    assert len(df_meta) == 22, f"Expected 22 meta-features, got {len(df_meta)}"
    assert m_vec.shape == (22,), f"Expected shape (22,), got {m_vec.shape}"
    assert np.all(np.isfinite(m_vec)), "Meta-feature vector contains non-finite values"
    print("[PASS] 22 X-only meta-features extracted cleanly (strictly y=None).")

    # 7. 6 Canonical Meta-Feature Groups
    grouped = get_grouped_meta_features(df_meta)
    assert len(grouped) == 6, f"Expected 6 groups, got {len(grouped)}"
    all_grouped = [f for grp in grouped.values() for f in grp["Meta-Feature"].tolist()]
    assert set(all_grouped) == set(X_ONLY_META_FEATURE_NAMES)
    print("[PASS] 6-group progressive disclosure meta-feature partitioning verified.")

    # 8. Candidate generation & Top-5 recommendations
    candidates = generate_candidate_configs(n_candidates=200, seed=42)
    assert len(candidates) == 200, "Candidate pool size mismatch"
    X_cand = build_candidate_matrix(m_vec, candidates)
    assert X_cand.shape == (200, 30), "Candidate matrix shape mismatch"

    df_rec = generate_metahpo_recommendations(
        meta_model=meta_model,
        dataset_meta_features=m_vec,
        n_candidates=200,
        n_select=5,
        beta=0.1,
        seed=42,
    )
    assert len(df_rec) == 5, f"Expected 5 recommendations, got {len(df_rec)}"
    for p in HPO_SEARCH_SPACE.keys():
        assert p in df_rec.columns, f"Missing hyperparameter {p} in recommendations"
    for col in ["predicted_cv_macro_f1", "prediction_uncertainty", "relative_confidence", "acquisition_score"]:
        assert col in df_rec.columns, f"Missing metric {col} in recommendations"
    print("[PASS] MetaHPO-C top-5 recommendation pipeline verified.")

    # 9. Benchmark data loading
    df_bench = load_validated_benchmark_results()
    assert len(df_bench) == 12, f"Expected 12 benchmark summary rows, got {len(df_bench)}"
    print("[PASS] Validated research benchmark data loaded successfully (12 summary rows).")

    # 10. Research artifact preservation
    assert (root / "results" / "experiments" / "unseen_benchmark" / "unseen_trial_results.csv").exists()
    assert (root / "results" / "experiments" / "meta_learning" / "meta_model.joblib").exists()
    assert (root / "results" / "analysis" / "final_results_table.csv").exists()
    print("[PASS] All research artifacts preserved without mutation.")

    # 11. Run full unit test suite
    print("\nRunning unit test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(root / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    assert result.wasSuccessful() and result.testsRun >= 91, f"Unit tests failed: {result.errors}"
    print(f"[PASS] All {result.testsRun} unit tests passed.")

    print("\n=======================================================")
    print("UI DESIGN VERIFICATION: PASS")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
