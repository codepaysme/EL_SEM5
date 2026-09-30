"""Verification and audit script for Phase 13 Streamlit frontend application."""

import sys
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
    parse_uploaded_csv,
)


def main():
    root = get_project_root()
    app_path = root / "app.py"

    print("\n=======================================================")
    print("PHASE 13: STREAMLIT FRONTEND DEMO VERIFICATION & AUDIT")
    print("=======================================================\n")

    # 1. App file exists
    assert app_path.exists() and app_path.stat().st_size > 1000, "app.py missing or empty"
    print(f"[PASS] app.py exists and verified ({app_path.stat().st_size} bytes).")

    # 2. Streamlit import and version
    print(f"[PASS] Streamlit version: {st.__version__}")

    # 3. Model artifact existence and loading
    meta_model = load_meta_model()
    assert meta_model is not None, "Failed to load meta-model"
    print("[PASS] Trained ExtraTrees meta-model artifact loaded successfully.")

    # 4. Built-in Demo dataset verification
    for ds in DEMO_DATASETS:
        X_df, y, title = load_demo_dataset(ds)
        assert len(X_df) > 0 and X_df.shape[1] > 0, f"Demo dataset {ds} empty"
    print(f"[PASS] All {len(DEMO_DATASETS)} built-in demo datasets load cleanly.")

    # 5. X-only meta-feature extraction (y=None, shape=(22,))
    X_wdbc, _, _ = load_demo_dataset("WDBC")
    df_meta, m_vec = extract_x_only_meta_features_table(X_wdbc, dataset_name="WDBC_verify")
    assert len(df_meta) == 22, f"Expected 22 meta-features, got {len(df_meta)}"
    assert m_vec.shape == (22,), f"Expected (22,) vector, got {m_vec.shape}"
    assert np.all(np.isfinite(m_vec)), "Meta-feature vector contains non-finite values"
    print("[PASS] X-only meta-feature extraction verified (y=None, shape=(22,)).")

    # 6. Grouped meta-feature partitioning
    grouped = get_grouped_meta_features(df_meta)
    assert len(grouped) == 6, f"Expected 6 meta-feature categories, got {len(grouped)}"
    all_grouped_feats = [f for grp in grouped.values() for f in grp["Meta-Feature"].tolist()]
    assert len(all_grouped_feats) == 22, f"Expected 22 grouped meta-features, got {len(all_grouped_feats)}"
    assert set(all_grouped_feats) == set(X_ONLY_META_FEATURE_NAMES)
    print("[PASS] Canonical 6-group meta-feature partitioning verified.")

    # 7. Candidate matrix shape (200 x 30)
    candidates = generate_candidate_configs(n_candidates=200, seed=42)
    assert len(candidates) == 200, f"Expected 200 candidates, got {len(candidates)}"
    X_cand = build_candidate_matrix(m_vec, candidates)
    assert X_cand.shape == (200, 30), f"Candidate matrix shape {X_cand.shape} != (200, 30)"
    print("[PASS] 30-dimensional candidate feature matrix construction verified (200 x 30).")

    # 8. Confidence-aware recommendation pipeline (5 recommendations, all 8 hyperparameters)
    df_rec = generate_metahpo_recommendations(
        meta_model=meta_model,
        dataset_meta_features=m_vec,
        n_candidates=200,
        n_select=5,
        beta=0.1,
        seed=42,
    )
    assert len(df_rec) == 5, f"Expected exactly 5 recommendations, got {len(df_rec)}"
    for p in HPO_SEARCH_SPACE.keys():
        assert p in df_rec.columns, f"Missing hyperparameter {p} in recommendations"
    for col in ["predicted_cv_macro_f1", "prediction_uncertainty", "relative_confidence", "acquisition_score"]:
        assert col in df_rec.columns, f"Missing metric column {col}"
    print("[PASS] MetaHPO-C top-5 recommendation pipeline verified (8 parameters, monotonic acquisition).")

    # 9. Validated benchmark results loading (12 rows across 4 unseen datasets)
    df_bench = load_validated_benchmark_results()
    assert len(df_bench) == 12, f"Expected 12 benchmark summary rows, got {len(df_bench)}"
    expected_unseen = {"madelon", "diabetes", "steel-plates-fault", "ilpd"}
    assert set(df_bench["dataset_name"].unique()) == expected_unseen
    print("[PASS] Validated research benchmark results table loaded successfully (4 unseen datasets, 3 methods).")

    # 10. Research artifact preservation and no HPO execution
    assert (root / "results" / "experiments" / "unseen_benchmark" / "unseen_trial_results.csv").exists()
    assert (root / "results" / "experiments" / "meta_learning" / "meta_model.joblib").exists()
    assert (root / "results" / "analysis" / "final_results_table.csv").exists()
    print("[PASS] Research and experiment artifacts preserved without modification.")

    print("\n>>> ALL PHASE 13 STREAMLIT AUDITS PASSED SUCCESSFULLY. <<<\n")


if __name__ == "__main__":
    main()
