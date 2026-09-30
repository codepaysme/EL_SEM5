# MetaHPO-C: Technical Methodology Notes & Viva Preparation

This document outlines key technical decisions, mathematical formulations, architectural trade-offs, and data-leakage safeguards implemented across Phases 1–11 of the **MetaHPO-C** project.

---

## 1. Meta-Feature Extraction & Leakage Prevention

### 1.1 Why X-Only Meta-Features?
- In genuine zero-shot or warm-started Hyperparameter Optimization on an unseen target dataset, the recommendation system must generate hyperparameter suggestions **before** ground-truth labels $y$ are evaluated.
- While target-dependent descriptive statistics (e.g., class balance, class entropy, number of classes) can describe a labeled historical dataset, using them at recommendation time for an unseen task creates a methodological flaw if the system claims to operate without ground-truth labels.
- Therefore, MetaHPO-C uses a strictly **22-dimensional X-only meta-feature vector** $\mathbf{m} \in \mathbb{R}^{22}$ computed exclusively from the input feature matrix $X$:
  1. `n_samples`
  2. `n_features`
  3. `feature_to_sample_ratio`
  4. `log_n_samples`
  5. `log_n_features`
  6. `numeric_feature_count`
  7. `numeric_feature_ratio`
  8. `mean_of_feature_means`
  9. `std_of_feature_means`
  10. `mean_of_feature_stds`
  11. `std_of_feature_stds`
  12. `mean_of_feature_min`
  13. `mean_of_feature_max`
  14. `missing_value_ratio`
  15. `features_with_missing_ratio`
  16. `categorical_feature_count`
  17. `categorical_feature_ratio`
  18. `mean_categorical_cardinality`
  19. `max_categorical_cardinality`
  20. `mean_absolute_correlation`
  21. `max_absolute_correlation`
  22. `mean_coefficient_of_variation`

### 1.2 Data Leakage Safeguards
1. **Historical vs. Unseen Separation:** 7 datasets were designated for historical HPO meta-training (`adult`, `bank-marketing`, `credit-g`, `phoneme`, `spambase`, `vehicle`, `wdbc`). 4 reserved datasets (`madelon`, `diabetes`, `steel-plates-fault`, `ilpd`) were strictly held out until Phase 10.
2. **Train/Test Splitting:** 80% train / 20% test stratified split (`random_state=42`) created prior to HPO. Test sets were locked and never passed to Optuna, CV folds, or candidate acquisition.
3. **Cross-Validation Preprocessing:** Missing-value imputation and feature standard scaling (`StandardScaler`) were fitted strictly inside each training fold of the 3-Fold Stratified CV, ensuring zero information leaked from validation folds.
4. **Single Test Evaluation:** Final test performance was evaluated exactly once per method per dataset using the best configuration selected by validation CV Macro-F1.

---

## 2. Meta-Learning Formulation & Response-Surface Surrogate

### 2.1 Learning Problem Formulation
- Rather than directly predicting a single hyperparameter vector $\boldsymbol{\theta}^* \in \mathbb{R}^8$ from meta-features $\mathbf{m}$ (which discards sub-optimal trial information and ignores multi-modal optima), MetaHPO-C frames meta-learning as a **supervised response-surface regression problem**:
  $$
  f: (\mathbf{m}, \boldsymbol{\theta}) \mapsto y \approx \text{CV Macro-F1}
  $$
- Input dimensionality: 22 dataset meta-features + 8 XGBoost hyperparameters = **30 predictive features**.
- Dataset size: 7 historical datasets $\times$ 20 HPO trials = **140 supervised training observations**.

### 2.2 Choice of Meta-Model: ExtraTreesRegressor
- Model configuration: `ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=-1)`.
- **Why Extremely Randomized Trees?**
  1. Non-linear interaction modeling between dataset meta-features and hyperparameters.
  2. Extreme randomization (random cut-point selection) provides strong regularization against overfitting on small meta-datasets ($N=140$).
  3. Fast parallel inference across large candidate configuration batches ($N=200$).
  4. Natural ensemble structure enabling variance-based uncertainty estimation across the $T=300$ individual decision trees.

### 2.3 Leave-One-Dataset-Out (LODO) Validation
- Standard $K$-fold CV is invalid for meta-learning because trials from the same dataset would appear in both train and test folds.
- LODO cross-validation holds out all 20 trials of an entire dataset, fits the meta-model on the remaining 120 trials (6 datasets), and evaluates predictions on the unseen task.
- **Results:** Mean within-dataset Spearman rank correlation $\bar{r}_s \approx +0.6941$ (ranging from $+0.5008$ on `vehicle` to $+0.8556$ on `spambase`), confirming strong candidate ranking capability despite dataset-level intercept shifts that produce a negative global $R^2$ ($-0.1912$).

---

## 3. Uncertainty Proxy & Confidence-Aware Candidate Acquisition

### 3.1 Ensemble Prediction Dispersion Uncertainty
- For any candidate pair $(\mathbf{m}, \boldsymbol{\theta})$, the 300 individual trees in the fitted ExtraTrees ensemble output predictions $p_1, p_2, \dots, p_{300}$.
- Expected score proxy:
  $$
  \hat{\mu}(\mathbf{m}, \boldsymbol{\theta}) = \frac{1}{T} \sum_{t=1}^{T} p_t(\mathbf{m}, \boldsymbol{\theta})
  $$
- Ensemble dispersion uncertainty proxy:
  $$
  u(\mathbf{m}, \boldsymbol{\theta}) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left(p_t(\mathbf{m}, \boldsymbol{\theta}) - \hat{\mu}(\mathbf{m}, \boldsymbol{\theta})\right)^2}
  $$
- **Empirical Validation:** On 140 LODO out-of-fold trials, $u$ exhibits a statistically significant positive rank correlation with absolute meta-model prediction error ($r_s = +0.1799, p = 0.0334$). Predictions with uncertainty above the median exhibited higher MAE (0.1160 vs. 0.1110).

### 3.2 Heuristic Upper-Confidence Acquisition
- Candidate pool: $N=200$ configurations uniformly sampled from the 8-parameter XGBoost search space using `seed=42`.
- Acquisition rule:
  $$
  \alpha(\boldsymbol{\theta}) = \hat{\mu}(\mathbf{m}, \boldsymbol{\theta}) + \beta \cdot u(\mathbf{m}, \boldsymbol{\theta})
  $$
- Trade-off parameter: $\beta = 0.1$.
- **Intuition:** Exploits high predicted validation performance while granting a modest bonus to regions of hyperparameter space where ensemble disagreement indicates higher exploration potential.

---

## 4. Benchmark Design & Budget Equalization

### 4.1 Equalized Evaluation Budget
- All three methods evaluated under an identical budget of **20 actual 3-fold Stratified CV evaluations**:
  - **Random Search:** 20 uniform random configurations evaluated sequentially.
  - **Optuna / TPE:** 20 Bayesian optimization trials driven by Tree-structured Parzen Estimator (`seed=42`).
  - **MetaHPO-C:** Top $K=5$ acquisition-ranked candidates enqueued as initial warm-start trials, followed by 15 adaptive Optuna/TPE exploration trials.
- Meta-model candidate generation and scoring ($200$ surrogate evaluations) take $< 0.1$ seconds and are **never** counted as real CV evaluations.

### 4.2 Statistical Power & Sample Size Nuances
- Independent experimental units: $n=4$ unseen datasets (`madelon`, `diabetes`, `steel-plates-fault`, `ilpd`).
- Single seed evaluation ($seed=42$) provides rigorous proof-of-concept verification.
- Paired statistical tests (Wilcoxon signed-rank and paired t-tests) correctly reflect limited power under $n=4$; non-significant p-values indicate small sample size rather than statistical equivalence.
