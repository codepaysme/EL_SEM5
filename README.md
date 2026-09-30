# MetaHPO-C: Confidence-Aware Meta-Learned Hyperparameter Optimization for Unseen Datasets

> **Academic ML Research Project (B.Tech 5th Semester / Applied ML)**  
> **Status:** Phases 1–14 Complete | Full Verification & Documentation Passed | 91 Automated Tests Passing

---

## 1. Executive Summary & Problem Statement

Hyperparameter Optimization (HPO) is critical for maximizing the predictive performance of gradient-boosted decision trees such as XGBoost. However, standard black-box HPO methods (e.g., Random Search, standard Bayesian Optimization / TPE) suffer from the **"cold-start" problem**: on every new dataset, the search begins from scratch with zero prior knowledge, requiring costly model training iterations across unpromising regions.

While **meta-learning** can transfer knowledge from prior HPO trajectories across historical datasets to recommend promising initial hyperparameters, existing meta-learning approaches often assume the unseen dataset is similar to the training distribution. When an unseen dataset is out-of-distribution (OOD) or dissimilar, unguided recommendations risk **negative transfer**, steering the search into suboptimal configurations.

**MetaHPO-C** introduces a **confidence-aware meta-learning framework** for hyperparameter optimization on unseen tabular classification datasets:
1. **Dataset Representation:** Computes 22 canonical $\mathbf{X}$-only meta-features strictly without target labels ($y=\text{None}$), preventing target leakage.
2. **Meta-Learned Surrogate:** Employs an `ExtraTreesRegressor` (300 trees) trained on historical dataset/HPO experience mapping $[22 \text{ meta-features} + 8 \text{ hyperparameters}] \to \text{CV Macro-F1}$.
3. **Uncertainty Proxy:** Quantifies ensemble prediction dispersion (tree disagreement standard deviation) as an empirical uncertainty proxy.
4. **Heuristic Candidate Acquisition:** Ranks $N=200$ candidate configurations via an upper-confidence acquisition score:
   $$\alpha = \hat{\mu} + \beta \cdot u \quad (\beta=0.1)$$
5. **Warm-Start Optimization:** Seeds the downstream optimization engine (Optuna TPE) with the top $K=5$ acquisition candidates, followed by 15 adaptive TPE trials (20 trials total budget).

---

## 2. System Pipeline Overview

```
                        UNSEEN TARGET DATASET (X)
                                    │
                                    ▼
                 22 X-Only Meta-Feature Extraction (y=None)
                                    │
                                    ▼
                      22-Dimensional Representation
                                    │
                                    ▼
       Candidate Generation (N=200 Uniform/Log-Uniform Configs, Seed=42)
                                    │
                                    ▼
           30-Dimensional Predictive Matrix [22 Meta-Features, 8 HPs]
                                    │
                                    ▼
                 Trained ExtraTrees Surrogate (300 Trees)
                                    │
                                    ▼
         Predicted CV Macro-F1 (μ̂)  +  Ensemble Dispersion Uncertainty (u)
                                    │
                                    ▼
            Confidence-Aware Acquisition Scoring (α = μ̂ + 0.1 · u)
                                    │
                                    ▼
                  Top-5 Warm-Start Configurations
                                    │
                                    ▼
      Downstream HPO Execution (5 Warm-Start Trials + 15 Adaptive TPE Trials)
```

---

## 3. Experimental Design

### A. Dataset Partitioning

The experimental setup enforces strict separation between meta-training and unseen evaluation datasets:

| Category | Count | Datasets Included | Role in Research |
|---|:---:|---|---|
| **Historical Datasets** | 7 | `adult`, `bank-marketing`, `credit-g`, `wdbc`, `spambase`, `vehicle`, `phoneme` | Used to generate 140 HPO training trials (20 per dataset) and train the meta-model. |
| **Reserved Unseen Datasets** | 4 | `madelon`, `diabetes`, `steel-plates-fault`, `ilpd` | Held out completely from meta-training; used solely for final benchmark evaluation. |
| **Interactive Demo Datasets** | 4 | `WDBC`, `Iris`, `Wine`, `Breast Cancer` | Built into the Streamlit dashboard for real-time meta-inference and feature inspection. |

### B. Meta-Feature Descriptors (22 Canonical X-Only Features)

All 22 meta-features are computed exclusively from the input feature matrix $\mathbf{X}$ with $y=\text{None}$:
1. **Dataset Size (5):** `n_samples`, `n_features`, `feature_to_sample_ratio`, `log_n_samples`, `log_n_features`
2. **Feature Composition (4):** `numeric_feature_count`, `numeric_feature_ratio`, `categorical_feature_count`, `categorical_feature_ratio`
3. **Distribution Statistics (7):** `mean_of_feature_means`, `std_of_feature_means`, `mean_of_feature_stds`, `std_of_feature_stds`, `mean_of_feature_min`, `mean_of_feature_max`, `mean_coefficient_of_variation`
4. **Missingness (2):** `missing_value_ratio`, `features_with_missing_ratio`
5. **Categorical Structure (2):** `mean_categorical_cardinality`, `max_categorical_cardinality`
6. **Correlation (2):** `mean_absolute_correlation`, `max_absolute_correlation`

### C. Hyperparameter Search Space (8 XGBoost Parameters)

| Hyperparameter | Search Range | Scale / Distribution | Type |
|---|:---:|:---:|:---:|
| `n_estimators` | $[50, 300]$ | Uniform | `int` |
| `max_depth` | $[2, 8]$ | Uniform | `int` |
| `learning_rate` | $[0.01, 0.30]$ | Log-Uniform | `float` |
| `subsample` | $[0.60, 1.00]$ | Uniform | `float` |
| `colsample_bytree` | $[0.60, 1.00]$ | Uniform | `float` |
| `min_child_weight` | $[1, 10]$ | Uniform | `int` |
| `gamma` | $[0.00, 2.00]$ | Uniform | `float` |
| `reg_lambda` | $[0.10, 10.00]$ | Log-Uniform | `float` |

*Note:* `reg_alpha` is explicitly not part of the search space.

### D. Benchmark Protocol & Baselines

All methods are evaluated under an equalized budget of **20 trials** using **Stratified 3-Fold Cross-Validation** (optimizing validation Macro-F1) on the 80% training split, followed by a single final evaluation on the 20% held-out test split:
- **Random Search:** 20 pseudo-random configurations sampled uniformly/log-uniformly ($\text{seed}=42$).
- **Standard Optuna TPE:** 20 trials of Tree-structured Parzen Estimator ($\text{seed}=42$).
- **MetaHPO-C:** 5 meta-recommended warm-start trials + 15 adaptive TPE trials ($\text{seed}=42$).

---

## 4. Confidence & Uncertainty Proxy Interpretation

MetaHPO-C utilizes the ensemble variance across the 300 decision trees in the `ExtraTreesRegressor`:
$$u(x) = \sqrt{\frac{1}{T} \sum_{t=1}^T \left( p_t(x) - \hat{\mu}(x) \right)^2}$$

> **Critical Methodological Notice:**  
> The uncertainty metric $u(x)$ is strictly an **empirical ensemble-dispersion uncertainty proxy** reflecting model disagreement across tree estimators. It is **NOT** a statistically calibrated probability, Bayesian posterior variance, or guarantee of prediction accuracy. The acquisition rule $\alpha = \hat{\mu} + 0.1 \cdot u$ is a heuristic candidate ranking strategy designed to balance exploitation (predicted performance) and exploration (prediction dispersion).

---

## 5. Strict Data Leakage Controls

To ensure rigorous scientific validity, the following 9 leakage controls are enforced across the entire codebase:
1. **Unseen Dataset Isolation:** The 4 unseen evaluation datasets were never included in meta-model training or Leave-One-Dataset-Out (LODO) cross-validation.
2. **Unlabeled Feature Extraction:** All meta-features extracted for unseen dataset recommendations explicitly pass $y=\text{None}$.
3. **Strict Train/Test Split:** 80% train / 20% test stratified split created prior to HPO; test data is touched exactly once.
4. **Fold-Isolated Preprocessing:** Imputation, scaling, and categorical encoding are fitted strictly within training folds during cross-validation.
5. **Isolated HPO Objective:** Cross-validation scores are computed only on the 80% training split.
6. **Final Model Fitting:** The best-discovered configuration is fitted once on the full 80% training set before evaluating on the 20% test split.
7. **LODO Fold Partitioning:** LODO evaluation isolates all 20 trials of a held-out dataset during meta-model cross-validation.
8. **Artifact Separation:** Historical meta-learning data and unseen benchmark results are persisted in separate directory hierarchies.
9. **Frozen Meta-Model:** The trained meta-model artifact (`meta_model.joblib`) is loaded read-only during inference without real-time retraining.

---

## 6. Authoritative Benchmark Results

Below are the exact held-out test Macro-F1 scores evaluated on the 20% test partition from `results/analysis/final_results_table.csv`:

| Dataset Name | Random Search | Optuna TPE | MetaHPO-C | Delta vs. Random | Delta vs. Optuna |
|---|:---:|:---:|:---:|:---:|:---:|
| `madelon` | 0.851918 | 0.853844 | **0.859590** | $+0.007672$ | $+0.005746$ |
| `diabetes` | 0.720742 | **0.729074** | 0.706834 | $-0.013908$ | $-0.022240$ |
| `steel-plates-fault` | **1.000000** | **1.000000** | **1.000000** | $0.000000$ | $0.000000$ |
| `ilpd` | 0.535714 | **0.558063** | 0.545062 | $+0.009347$ | $-0.013001$ |

### Scientific Conclusion & Findings
- **Feasibility Demonstrated:** MetaHPO-C demonstrates that historical dataset/HPO experience can effectively warm-start HPO on unseen datasets without target leakage.
- **Dataset-Dependent Outcomes:** MetaHPO-C achieved the highest held-out test Macro-F1 on `madelon` (0.8596), tied on `steel-plates-fault` (1.0000), and performed below Optuna TPE on `diabetes` (0.7068) and `ilpd` (0.5451).
- **Statistical Scope:** The benchmark was evaluated across 4 unseen datasets with a fixed seed ($\text{seed}=42$). While paired $t$-tests ($p=0.893$ vs Random, $p=0.328$ vs Optuna) do not demonstrate universal statistical superiority, the results provide solid empirical evidence of meta-learning feasibility.
- **Uncertainty Proxy Correlation:** During LODO validation across 140 historical trials, the ensemble-dispersion uncertainty proxy exhibited a modest positive correlation ($r_s = +0.1799, p = 0.0334$) with meta-model prediction error.

---

## 7. Setup & Installation

### A. Environment Setup (Windows PowerShell)

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\Activate.ps1
# (If script execution is disabled: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
# Alternatively, invoke .venv\Scripts\python.exe directly

# 3. Install required dependencies
python -m pip install -r requirements.txt
```

---

## 8. Running the Application & Verification

### A. Launch Interactive Research Dashboard

```powershell
python -m streamlit run app.py
```
> The dashboard runs locally at `http://localhost:8501`. It demonstrates real-time X-only meta-feature extraction, 200-candidate surrogate scoring, top-5 recommendation generation, acquisition landscape visualizations, and displays the authoritative pre-computed benchmark results.

### B. Run Complete Automated Test Suite (91 Tests)

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### C. Run Standalone Verification Scripts

```powershell
# Phase 10: Unseen Dataset Benchmark Verification
python scripts/verify_unseen_benchmark.py

# Phase 11: Statistical Analysis & Figures Verification
python scripts/verify_final_analysis.py

# Phase 12: Technical Report Verification
python scripts/verify_final_report.py

# Phase 13: Streamlit UI Functionality Verification
python scripts/verify_streamlit_app.py

# Phase 14: End-to-End Final Project Verification
python scripts/verify_final_project.py
```

---

## 9. Project Repository Structure

```
MetaHPO/
│
├── README.md                           # Main project documentation and overview
├── requirements.txt                    # Project dependencies (Python 3.11+)
├── .gitignore                          # Version control exclusion rules
├── app.py                              # Streamlit academic research dashboard
│
├── config/
│   └── datasets.yaml                   # Dataset registry (historical & unseen)
│
├── data/
│   ├── raw/                            # Raw OpenML dataset caches
│   ├── processed/                      # Preprocessed feature matrices
│   └── metadata/                       # Dataset meta-features and statistics
│
├── docs/
│   ├── final_report.md                 # Complete 19-section academic technical report
│   ├── results_tables.md               # 8 comprehensive experimental results tables
│   ├── report_figures.md               # Publication figure catalog and descriptions
│   ├── methodology_notes.md            # Detailed mathematical and architectural notes
│   ├── demo_and_viva_guide.md          # 3-min demo script, 5-min tech talk, 25 viva Q&As
│   ├── project_cheat_sheet.md          # One-page quick reference sheet for viva
│   └── final_project_status.md         # Phase-by-phase completion record
│
├── notebooks/                          # Jupyter exploration notebooks
│
├── scripts/
│   ├── run_historical_hpo.py           # Historical HPO experiment runner
│   ├── run_unseen_benchmark.py         # 4-dataset unseen benchmark runner
│   ├── run_final_analysis.py           # Statistical analysis and figure generator
│   ├── verify_meta_features.py         # Meta-feature verification script
│   ├── verify_meta_dataset.py          # Meta-dataset verification script
│   ├── verify_meta_model.py            # Meta-model LODO verification script
│   ├── verify_uncertainty.py           # Uncertainty estimation verification script
│   ├── verify_acquisition.py           # Candidate acquisition verification script
│   ├── verify_unseen_benchmark.py      # Benchmark audit script
│   ├── verify_final_analysis.py        # Analysis audit script
│   ├── verify_final_report.py          # Final report audit script
│   ├── verify_streamlit_app.py         # Streamlit UI verification script
│   └── verify_final_project.py         # End-to-end master verification script
│
├── src/
│   ├── data/                           # Dataset loading, validation, splitting & preprocessing
│   ├── features/                       # 22 X-only and 4 labeled meta-feature extractors
│   ├── hpo/                            # Optuna + XGBoost HPO engine & CV evaluation
│   ├── meta_learning/                  # Supervised meta-dataset & ExtraTrees meta-model
│   ├── confidence/                     # Dispersion uncertainty & acquisition scoring
│   ├── evaluation/                     # Statistical analysis, metrics, and plotting
│   ├── ui/                             # Streamlit dashboard helper functions & tables
│   └── utils/                          # Logging and path utilities
│
├── tests/                              # 91 unit and integration tests
│   ├── test_loader.py
│   ├── test_meta_features.py
│   ├── test_hpo.py
│   ├── test_historical_experiment.py
│   ├── test_meta_dataset.py
│   ├── test_meta_model.py
│   ├── test_uncertainty.py
│   ├── test_acquisition.py
│   ├── test_unseen_benchmark.py
│   ├── test_final_analysis.py
│   └── test_app_helpers.py
│
└── results/
    ├── experiments/
    │   ├── historical/                 # 140 historical HPO trial trajectories
    │   ├── meta_learning/              # Supervised meta-dataset & trained meta-model
    │   └── unseen_benchmark/           # 240 benchmark trial records & efficiency curves
    └── analysis/
        ├── final_results_table.csv     # Authoritative benchmark results table
        ├── hypothesis_tests.csv        # Paired t-test and Wilcoxon signed-rank results
        ├── warm_start_diagnostics.csv  # Trial 1-5 warm-start diagnostic metrics
        └── figures/                    # 6 publication figures (300 DPI)
```
