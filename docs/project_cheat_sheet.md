# MetaHPO-C: One-Page Project Cheat Sheet

> **Quick-Reference Summary for Viva Defense & Review**

---

### Core Identity & Problem
- **Project Name:** MetaHPO-C (Confidence-Aware Meta-Learned Hyperparameter Optimization for Unseen Datasets)
- **Target Model:** `XGBClassifier` (XGBoost)
- **Primary Problem:** Cold-start inefficiency in tabular classification HPO
- **Solution:** Meta-learned response-surface surrogate + ensemble-dispersion uncertainty-aware candidate warm-starting

---

### Key System Specifications

| Component | Specification | Details |
|---|---|---|
| **Historical Datasets** | 7 OpenML Datasets | `adult`, `bank-marketing`, `credit-g`, `wdbc`, `spambase`, `vehicle`, `phoneme` |
| **Historical HPO Budget** | 20 Trials / Dataset | 140 Supervised Meta-Observations Total |
| **Dataset Meta-Features** | 22 Canonical Descriptors | **Strictly $\mathbf{X}$-only** ($y=\text{None}$); 6 functional groups |
| **Hyperparameters** | 8 XGBoost Parameters | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda` (`reg_alpha` omitted) |
| **Meta-Model Input** | 30 Features | $[22 \text{ meta-features}, 8 \text{ hyperparameters}]$ |
| **Meta-Model Architecture** | `ExtraTreesRegressor` | 300 Trees, `min_samples_leaf=2`, `max_features=1.0`, `random_state=42` |
| **Target Variable** | Validation CV Macro-F1 | Mean Macro-F1 across 3 Stratified Cross-Validation folds |
| **Candidate Pool** | $N=200$ Configurations | Uniform and Log-Uniform sampling ($\text{seed}=42$) |
| **Uncertainty Proxy** | Tree Prediction Dispersion | $u(x) = \text{std}(p_1(x), \dots, p_T(x))$ across 300 trees |
| **Acquisition Formula** | Upper-Confidence Heuristic | $\alpha(x) = \hat{\mu}(x) + \beta \cdot u(x)$ with $\beta = 0.1$ |
| **Warm-Start Count** | Top $K=5$ Configurations | Top 5 ranked candidates evaluated in Trials 1–5 |
| **Downstream Optimizer** | Optuna TPE | 15 Subsequent Adaptive Trials (Trials 6–20; total 20 trials) |
| **Unseen Evaluation Datasets** | 4 Reserved Datasets | `madelon`, `diabetes`, `steel-plates-fault`, `ilpd` (never used in meta-training) |
| **Evaluation Splits** | 80% Train / 20% Test | Stratified split; test partition evaluated strictly once at the end |
| **Reproducibility Seed** | Fixed Seed = 42 | Deterministic runs across data loading, HPO, and candidate sampling |

---

### Authoritative Final Test Results (20% Held-Out Split)

| Dataset | Random Search | Optuna TPE | MetaHPO-C | Outcome Summary |
|---|:---:|:---:|:---:|---|
| `madelon` | 0.8519 | 0.8538 | **0.8596** | MetaHPO-C achieves top test Macro-F1 (+0.0057 vs Optuna) |
| `diabetes` | 0.7207 | **0.7291** | 0.7068 | Optuna leads; meta-recommendations suffered slight mismatch |
| `steel-plates-fault` | **1.0000** | **1.0000** | **1.0000** | All three methods reach identical ceiling performance |
| `ilpd` | 0.5357 | **0.5581** | 0.5451 | Optuna leads; MetaHPO-C beats Random Search |

---

### Methodological Controls & Scientific Caveats

1. **Zero Target Leakage:** Meta-features for unseen datasets are extracted with $y=\text{None}$.
2. **Dataset Isolation:** Reserved unseen datasets were completely excluded from meta-model fitting and LODO cross-validation.
3. **No Retraining during Demo:** Streamlit dashboard loads pre-computed `meta_model.joblib` artifact for sub-50ms inference.
4. **Uncertainty Interpretation:** The ensemble standard deviation is an **empirical dispersion proxy**, not a calibrated Bayesian probability.
5. **No Universal Superiority Claim:** Performance is dataset-dependent; the findings prove **feasibility and competitive warm-starting**, not universal dominance.
