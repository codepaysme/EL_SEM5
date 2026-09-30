r"""MetaHPO-C: Confidence-Aware Meta-Learned Hyperparameter Optimization for Unseen Datasets.

Streamlit Academic Research Demonstration Dashboard.
Designed with Progressive Disclosure: Simple landing experience with technical depth on demand.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.confidence.acquisition import (
    HPO_SEARCH_SPACE,
    build_candidate_matrix,
    compute_acquisition_score,
    generate_candidate_configs,
    score_candidates,
)
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

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Academic Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MetaHPO-C | Research Product",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Restrained Academic Custom CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 0.8rem;
    }
    .pipeline-badge {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        border: 1px solid #E2E8F0;
    }
    .rec-top-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }
    .param-badge {
        display: inline-block;
        background-color: #F8FAFC;
        color: #334155;
        padding: 0.22rem 0.48rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.82rem;
        margin: 0.18rem;
        border: 1px solid #E2E8F0;
    }
    .setup-line {
        color: #64748B;
        font-size: 0.88rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Caching Functions
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_meta_model() -> Any:
    """Cache the loaded ExtraTrees meta-model artifact."""
    return load_meta_model()


@st.cache_data(show_spinner=False)
def get_cached_benchmark_results() -> pd.DataFrame:
    """Cache the validated research benchmark results table."""
    try:
        return load_validated_benchmark_results()
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Minimal Sidebar (Dataset Selector & App Identifier)
# -----------------------------------------------------------------------------
st.sidebar.title("🧬 MetaHPO-C")
st.sidebar.caption("Confidence-Aware Meta-Learned HPO")
st.sidebar.divider()

st.sidebar.markdown("**Dataset Selection**")
sidebar_input_choice = st.sidebar.selectbox(
    "Choose dataset source:",
    DEMO_DATASETS + ["Upload Custom CSV"],
    index=0,
    label_visibility="collapsed",
)

uploaded_file_obj = None
target_col_name = ""

if sidebar_input_choice == "Upload Custom CSV":
    uploaded_file_obj = st.sidebar.file_uploader(
        "Upload CSV file:",
        type=["csv"],
        help="Upload a tabular CSV file.",
    )
    target_col_name = st.sidebar.text_input(
        "Target column (optional):",
        value="",
        help="Column to exclude from feature extraction.",
    ).strip()

st.sidebar.divider()
st.sidebar.markdown("**Mode:** Demonstration")
st.sidebar.caption("MetaHPO-C v1.0 • Academic ML Research")


# -----------------------------------------------------------------------------
# Top-Level Navigation Tabs
# -----------------------------------------------------------------------------
tab_explore, tab_results, tab_about = st.tabs([
    "🚀 Explore",
    "📊 Results",
    "ℹ️ About",
])


# =============================================================================
# TAB 1: 🚀 Explore (Primary Landing Experience)
# =============================================================================
with tab_explore:
    # 1. Compact Hero
    st.markdown('<div class="main-title">MetaHPO-C</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Confidence-Aware Meta-Learned Hyperparameter Optimization</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline-badge">Dataset Fingerprint &nbsp;→&nbsp; Meta-Learning &nbsp;→&nbsp; Recommendations</div>',
        unsafe_allow_html=True,
    )
    st.caption("Use prior experience from historical datasets to identify promising hyperparameters for a new dataset.")
    st.caption("Research demonstration • Recommendations are surrogate-based estimates and not guaranteed.")

    # Compact Research Setup Line
    st.markdown(
        '<div class="setup-line"><b>Research Setup:</b> 7 historical datasets · 140 trials · 22 X-only features · 300-tree surrogate</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View research configuration ▾", expanded=False):
        c_conf1, c_conf2, c_conf3, c_conf4, c_conf5 = st.columns(5)
        c_conf1.metric("Historical Data", "7 Datasets", "140 HPO Trials")
        c_conf2.metric("Meta-Features", "22 Descriptors", "X-Only (y=None)")
        c_conf3.metric("Meta-Model", "ExtraTrees", "300 Trees")
        c_conf4.metric("Candidate Pool", "200 Configs", "Top-5 Selected")
        c_conf5.metric("Acquisition Rule", "μ̂ + 0.1 · u", "Macro-F1 Target")

    st.divider()

    # 2. Dataset Loading & Compact Overview
    X_df: Optional[pd.DataFrame] = None
    y_series: Optional[pd.Series] = None
    dataset_title = "Target Dataset"

    try:
        if sidebar_input_choice != "Upload Custom CSV":
            X_df, y_series, dataset_title = load_demo_dataset(sidebar_input_choice)
        else:
            if uploaded_file_obj is not None:
                X_df, y_series = parse_uploaded_csv(
                    uploaded_file_obj,
                    target_column=target_col_name if target_col_name else None,
                )
                dataset_title = f"Uploaded ({uploaded_file_obj.name})"
            else:
                st.info("Please upload a CSV file via the sidebar to proceed.")

    except Exception as ex:
        st.error(f"Error loading dataset: {ex}")
        X_df = None

    if X_df is not None:
        stats = get_dataset_summary_stats(X_df)

        st.subheader(f"Dataset Overview: {dataset_title}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples (N)", f"{stats['n_samples']:,}")
        m2.metric("Features (P)", f"{stats['n_features']}")
        m3.metric("Numeric Features", f"{stats['numeric_feature_count']}")
        m4.metric("Missing Ratio", f"{stats['missing_value_ratio']:.2%}")

        # Progressive Disclosure: Dataframe Preview
        with st.expander("View dataset preview ▾", expanded=False):
            st.dataframe(X_df.head(5), width="stretch")

        # Progressive Disclosure: 22 Meta-Features
        with st.expander("Dataset fingerprint (22 X-only meta-features) ▾", expanded=False):
            st.markdown(
                r"All 22 descriptors are computed exclusively from the input feature matrix $\mathbf{X}$ "
                r"with $y=\text{None}$, guaranteeing zero target label leakage."
            )
            with st.spinner("Extracting 22 X-only meta-features..."):
                df_meta_table, m_array = extract_x_only_meta_features_table(
                    X_df, dataset_name=dataset_title
                )
                grouped_meta = get_grouped_meta_features(df_meta_table)

            tab_grp, tab_all = st.tabs(["Categorized Fingerprint", "Complete 22-Feature Table"])
            with tab_grp:
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.markdown("**1. Dataset Size (5)**")
                    st.dataframe(grouped_meta["Dataset Size"], width="stretch", hide_index=True)
                    st.markdown("**2. Feature Composition (4)**")
                    st.dataframe(grouped_meta["Feature Composition"], width="stretch", hide_index=True)
                    st.markdown("**3. Missingness (2)**")
                    st.dataframe(grouped_meta["Missingness"], width="stretch", hide_index=True)
                with g_col2:
                    st.markdown("**4. Distribution Statistics (7)**")
                    st.dataframe(grouped_meta["Distribution Statistics"], width="stretch", hide_index=True)
                    st.markdown("**5. Categorical Structure (2)**")
                    st.dataframe(grouped_meta["Categorical Structure"], width="stretch", hide_index=True)
                    st.markdown("**6. Correlation (2)**")
                    st.dataframe(grouped_meta["Correlation"], width="stretch", hide_index=True)
            with tab_all:
                st.dataframe(df_meta_table, width="stretch", hide_index=True)

        st.divider()

        # 3. Primary Visual Action: Recommendation Generation
        st.subheader("Generate Recommendations")
        st.caption("Evaluates 200 candidate hyperparameter configurations through the trained response-surface surrogate.")

        if st.button("⚡ Generate MetaHPO-C Recommendations", type="primary"):
            try:
                with st.spinner("Scoring 200 candidate configurations with ExtraTrees meta-model..."):
                    meta_model = get_cached_meta_model()
                    df_rec = generate_metahpo_recommendations(
                        meta_model=meta_model,
                        dataset_meta_features=m_array,
                        n_candidates=200,
                        n_select=5,
                        beta=0.1,
                        seed=42,
                    )
                    st.session_state["recommendations"] = df_rec
                    st.session_state["m_array"] = m_array

            except FileNotFoundError as fnf_err:
                st.error(f"Meta-model artifact missing: {fnf_err}")
            except Exception as err:
                st.error(f"Recommendation computation failed: {err}")

        # 4. Recommendation Display (Visual Centerpiece with Progressive Disclosure)
        if "recommendations" in st.session_state:
            df_rec = st.session_state["recommendations"]

            st.subheader("Recommended Configurations")
            st.caption("Top candidate configurations selected using predicted performance and an uncertainty-aware acquisition score.")

            # Top 3 Cards
            card_col1, card_col2, card_col3 = st.columns(3)
            medals = ["🏆 #1 Top Candidate", "🥈 #2 Candidate", "🥉 #3 Candidate"]
            cols = [card_col1, card_col2, card_col3]

            for i in range(min(3, len(df_rec))):
                row = df_rec.iloc[i]
                with cols[i]:
                    st.markdown(f"### {medals[i]}")
                    st.metric("Predicted CV Macro-F1", f"{row['predicted_cv_macro_f1']:.4f}")
                    st.caption(f"Uncertainty: {row['prediction_uncertainty']:.4f} &nbsp;|&nbsp; Acquisition: {row['acquisition_score']:.4f}")

                    # Progressive disclosure of parameters
                    with st.expander("View configuration ▾", expanded=False):
                        st.markdown(f"**Relative Confidence:** `{row['relative_confidence']:.4f}`")
                        param_html = " ".join([
                            f"<span class='param-badge'><b>{p}</b>: {int(row[p]) if HPO_SEARCH_SPACE[p][2] == 'int' else (f'{row[p]:.4f}' if isinstance(row[p], float) else row[p])}</span>"
                            for p in HPO_SEARCH_SPACE.keys()
                            if p in row
                        ])
                        st.markdown(param_html, unsafe_allow_html=True)

            # Recommendations #4 & #5 in a compact table
            if len(df_rec) > 3:
                st.markdown("#### Additional Warm-Start Recommendations (#4 & #5)")
                df_rest = df_rec.iloc[3:5][["Rank", "predicted_cv_macro_f1", "prediction_uncertainty", "relative_confidence", "acquisition_score"]]
                st.dataframe(df_rest, width="stretch", hide_index=True)
                with st.expander("View configurations for #4 & #5 ▾", expanded=False):
                    for idx in [3, 4]:
                        if idx < len(df_rec):
                            r_row = df_rec.iloc[idx]
                            st.markdown(f"**Recommendation #{r_row['Rank']}:**")
                            p_html = " ".join([
                                f"<span class='param-badge'><b>{p}</b>: {int(r_row[p]) if HPO_SEARCH_SPACE[p][2] == 'int' else (f'{r_row[p]:.4f}' if isinstance(r_row[p], float) else r_row[p])}</span>"
                                for p in HPO_SEARCH_SPACE.keys()
                                if p in r_row
                            ])
                            st.markdown(p_html, unsafe_allow_html=True)

            st.divider()

            # 5. Simple "Why?" Explanation
            st.subheader("Why these recommendations?")
            st.markdown(
                "MetaHPO-C compares the new dataset's fingerprint with historical HPO experience, "
                "predicts how candidate configurations may perform, and balances predicted performance with uncertainty."
            )
            with st.expander("How does this work? ▾", expanded=False):
                st.markdown(
                    r"""
                    **Acquisition Formula:**
                    $$\alpha = \hat{\mu} + 0.1 \cdot u$$

                    - **Exploitation ($\hat{\mu}$):** Rewards configurations with high predicted validation CV Macro-F1.
                    - **Exploration ($u$):** Incorporates ensemble dispersion (tree disagreement standard deviation) to explore under-sampled regions.
                    - **Methodological Notice:** The uncertainty quantity $u$ is an empirical dispersion proxy, not a calibrated Bayesian probability.
                    """
                )

            # 6. Candidate Landscape (Progressive Disclosure)
            with st.expander("Explore candidate landscape (200 candidates) ▾", expanded=False):
                st.caption("200 candidate configurations evaluated by the ExtraTrees meta-model.")
                col_v1, col_v2 = st.columns(2)

                with col_v1:
                    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
                    meta_model = get_cached_meta_model()
                    all_candidates = generate_candidate_configs(n_candidates=200, seed=42)

                    X_all = build_candidate_matrix(st.session_state["m_array"], all_candidates)
                    df_scored = score_candidates(meta_model, X_all, all_candidates)
                    df_scored["predicted_cv_macro_f1"] = df_scored["predicted_score"]
                    df_scored["acquisition_score"] = compute_acquisition_score(
                        df_scored["predicted_score"].to_numpy(),
                        df_scored["prediction_uncertainty"].to_numpy(),
                        beta=0.1,
                    )

                    ax.scatter(
                        df_scored["predicted_cv_macro_f1"],
                        df_scored["prediction_uncertainty"],
                        c="#1f77b4",
                        alpha=0.45,
                        s=35,
                        label="Candidate Pool (N=200)",
                        edgecolors="none",
                    )
                    ax.scatter(
                        df_rec["predicted_cv_macro_f1"],
                        df_rec["prediction_uncertainty"],
                        c="#d62728",
                        s=80,
                        label="Top-5 Selected",
                        edgecolors="black",
                        zorder=5,
                    )
                    ax.set_xlabel(r"Predicted CV Macro-F1 ($\hat{\mu}$)")
                    ax.set_ylabel(r"Ensemble Uncertainty Proxy ($u$)")
                    ax.set_title("Candidate Exploration vs. Exploitation Landscape")
                    ax.legend(frameon=True, loc="upper left", fontsize=8.5)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col_v2:
                    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
                    ranks = [f"Rank {r}" for r in df_rec["Rank"]]
                    y_pos = np.arange(len(ranks))
                    ax.barh(y_pos, df_rec["acquisition_score"], color="#d62728", edgecolor="black", height=0.6)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(ranks)
                    ax.invert_yaxis()
                    ax.set_xlabel(r"Acquisition Score ($\alpha = \hat{\mu} + 0.1 \cdot u$)")
                    ax.set_title("Top-5 Acquisition Scores")
                    for i, v in enumerate(df_rec["acquisition_score"]):
                        ax.text(v * 0.90, i, f"{v:.4f}", color="white", fontweight="bold", va="center", ha="right")
                    ax.grid(axis="x", linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                st.markdown("#### Configuration Detail Inspector")
                selected_rank_label = st.selectbox(
                    "Select a recommendation to inspect:",
                    [f"Recommendation #{r}" for r in df_rec["Rank"]],
                    index=0,
                )
                rank_idx = int(selected_rank_label.split("#")[1]) - 1
                selected_row = df_rec.iloc[rank_idx]
                st.json({p: selected_row[p] for p in HPO_SEARCH_SPACE.keys() if p in selected_row})


# =============================================================================
# TAB 2: 📊 Results (Benchmark Comparison)
# =============================================================================
with tab_results:
    st.header("Validated Research Results")
    st.caption("Authoritative held-out test results from the completed 4-dataset unseen benchmark (Phases 10 & 11).")

    df_benchmark = get_cached_benchmark_results()

    if not df_benchmark.empty:
        # Primary Visualization: Held-out Test Macro-F1
        fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
        datasets = ["diabetes", "ilpd", "madelon", "steel-plates-fault"]
        methods = ["random_search", "optuna_tpe", "metahpo_c"]
        method_labels = {
            "random_search": "Random Search",
            "optuna_tpe": "Optuna TPE",
            "metahpo_c": "MetaHPO-C",
        }
        colors = {"random_search": "#7f7f7f", "optuna_tpe": "#1f77b4", "metahpo_c": "#d62728"}

        x = np.arange(len(datasets))
        width = 0.25

        for i, method in enumerate(methods):
            scores = []
            for ds in datasets:
                row = df_benchmark[(df_benchmark["dataset_name"] == ds) & (df_benchmark["method"] == method)]
                if not row.empty:
                    scores.append(float(row["test_macro_f1"].iloc[0]))
                else:
                    scores.append(0.0)

            offset = (i - 1) * width
            ax.bar(x + offset, scores, width, label=method_labels[method], color=colors[method], edgecolor="black")

        ax.set_ylabel("Held-Out Test Macro-F1")
        ax.set_title("Held-Out Test Macro-F1 across 4 Reserved Unseen Datasets")
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("-", " ").title() for d in datasets])
        ax.set_ylim(0.0, 1.15)
        ax.legend(frameon=True, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Compact Benchmark Table
        st.markdown("#### Benchmark Summary Table")
        st.dataframe(df_benchmark[["dataset_name", "method", "best_cv_macro_f1", "test_macro_f1", "test_accuracy", "total_runtime_seconds"]], width="stretch", hide_index=True)

        # Progressive Disclosure: Full Benchmark Details
        with st.expander("Explore benchmark details & analysis ▾", expanded=False):
            st.markdown("#### What the Experiments Show")
            st.markdown(
                """
                - **Dataset-Dependent Outcomes:** Performance varies across the four unseen datasets.
                - **Madelon:** MetaHPO-C achieved the highest held-out test Macro-F1 ($0.8596$), finding high-performing regions early.
                - **Steel-Plates-Fault:** All three methods reached the identical ceiling performance ($1.0000$).
                - **Diabetes & ILPD:** Standard Optuna/TPE discovered higher test configurations ($0.7291$ and $0.5581$).
                - **Statistical Scope:** Evaluated across 4 unseen datasets with fixed seed ($\text{seed}=42$), indicating empirical feasibility rather than universal dominance.
                """
            )
            st.info(
                "**Conclusion:** The benchmark demonstrates feasibility of meta-learned warm-starting on unseen datasets. "
                "MetaHPO-C does not claim universal superiority over standard HPO baselines."
            )

    else:
        st.warning("Benchmark results artifact not found.")


# =============================================================================
# TAB 3: ℹ️ About (Methodology & Architecture)
# =============================================================================
with tab_about:
    st.header("About MetaHPO-C")
    st.caption("Theoretical foundation, system architecture, leakage controls, and research scope.")

    with st.expander("What is MetaHPO-C? ▾", expanded=False):
        st.markdown(
            """
            **MetaHPO-C** is a confidence-aware meta-learning framework designed to address the **cold-start problem** in Hyperparameter Optimization (HPO).
            By mapping dataset-level structural properties to validation performance across historical HPO trajectories, MetaHPO-C recommends high-quality initial configurations for XGBoost on unseen tabular classification datasets.
            """
        )

    with st.expander("How the System Works ▾", expanded=False):
        st.code(
            """
Historical HPO Trajectories (140 Trials across 7 Datasets)
       ↓
Dataset Fingerprint (22 X-Only Meta-Features, y=None)
       ↓
Meta-Learning Surrogate (ExtraTreesRegressor, 300 Trees)
       ↓
Candidate Pool Evaluation (200 Uniform/Log-Uniform Configs)
       ↓
Uncertainty-Aware Acquisition (α = μ̂ + 0.1 · u)
       ↓
Top-5 Warm-Start Recommendations for Downstream HPO
            """,
            language="text",
        )

    with st.expander("Why Meta-Learning? (Cold-Start vs. Warm-Start) ▾", expanded=False):
        st.markdown(
            """
            Traditional HPO algorithms (such as Random Search or standard Bayesian Optimization) start from scratch for every new dataset.
            When model training is computationally demanding, early cold-start iterations waste evaluation budget on unpromising regions.
            Meta-learning initializes the search in regions known to perform well on similar datasets.
            """
        )

    with st.expander("Understanding the Uncertainty Proxy ▾", expanded=False):
        st.markdown(
            r"""
            The uncertainty quantity $u(x)$ is calculated as the standard deviation across individual tree predictions in the `ExtraTreesRegressor` ensemble:
            $$u(x) = \text{std}(p_1(x), \dots, p_T(x))$$

            **Important Caveat:** This is strictly an **empirical ensemble-dispersion uncertainty proxy**, not a calibrated Bayesian posterior probability or guarantee of accuracy.
            """
        )

    with st.expander("Research Integrity & Leakage Controls ▾", expanded=False):
        st.markdown(
            """
            - [x] **Zero Target Leakage:** Meta-features for unseen datasets are extracted with $y=\text{None}$.
            - [x] **Unseen Dataset Isolation:** Reserved benchmark datasets (`madelon`, `diabetes`, `steel-plates-fault`, `ilpd`) were never used for meta-model training.
            - [x] **Fold-Isolated Preprocessing:** Preprocessing transformers are fitted strictly inside CV folds.
            - [x] **Single Test Evaluation:** Held-out test partitions are evaluated exactly once.
            - [x] **Read-Only Inference:** Pre-trained `meta_model.joblib` artifact is loaded without real-time retraining.
            """
        )

    with st.expander("Limitations & Research Scope ▾", expanded=False):
        st.markdown(
            """
            - **Meta-Training Size:** Trained on 140 trials across 7 OpenML classification datasets.
            - **Benchmark Scale:** Evaluated across 4 unseen datasets with a fixed seed ($\text{seed}=42$).
            - **Search Space:** Compact 8-parameter space tailored for `XGBClassifier`.
            - **Uncertainty Calibration:** Heuristic dispersion proxy rather than exact Bayesian inference.
            """
        )

    with st.expander("Experimental Extension: Fingerprint-Guided Search-Space Pruning ▾", expanded=False):
        st.markdown(
            """
            *Status: Experimental Extension*

            The project also investigates whether dataset fingerprints can be used to derive conservative, task-specific hyperparameter search spaces before candidate generation.
            This extension is preserved for future investigation and is not part of the primary validated recommendation flow.
            """
        )
