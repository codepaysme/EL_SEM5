# MetaHPO-C: Demonstration & Viva Defense Guide

> **Project Title:** MetaHPO-C: Confidence-Aware Meta-Learned Hyperparameter Optimization for Unseen Datasets  
> **Target Audience:** Examiners, Reviewers, and B.Tech Project Viva Panel  
> **Document Purpose:** Practical 3-minute demo script, 5-minute technical explanation, and 25 comprehensive Viva Q&As.

---

# Part A: 3-Minute Live Demo Script

Follow this structured sequence during your live project presentation:

| Time | Step | Action in Streamlit UI | Key Points to Speak to the Panel |
|:---:|---|---|---|
| **0:00 - 0:30** | **1. System Header & Config** | Open dashboard at `http://localhost:8501`. Show the top header and the 5 research metric cards. | *"MetaHPO-C solves the cold-start problem in hyperparameter optimization by learning a surrogate model from 140 historical HPO trials across 7 OpenML datasets. It recommends initial XGBoost configurations for unseen datasets using 22 X-only meta-features and an ensemble-dispersion uncertainty proxy."* |
| **0:30 - 1:00** | **2. Target Dataset & Meta-Features** | Select **WDBC** from the demo dropdown. Show the 5 dataset summary metric cards (569 samples, 30 features, 0% missing). Open the **22 X-Only Meta-Features** expander. | *"Notice that all 22 meta-features are extracted strictly without target labels ($y=\text{None}$). This ensures zero target leakage. The features cover dataset dimensionality, statistical moments, missingness, categorical cardinality, and feature correlations."* |
| **1:00 - 1:45** | **3. Live Candidate Recommendation** | Click the **⚡ Generate MetaHPO-C Recommendations** button. Expand **Recommendation #1**. | *"In real-time, the app samples 200 candidate hyperparameter configurations, evaluates them with the pre-trained 300-tree ExtraTrees surrogate, calculates tree-disagreement uncertainty $u$, and computes the acquisition score $\alpha = \hat{\mu} + 0.1 \cdot u$. The top 5 configurations are selected to warm-start downstream HPO."* |
| **1:45 - 2:15** | **4. Acquisition Landscape** | Scroll to the **Candidate Exploration vs. Exploitation Landscape** scatter plot and horizontal bar chart. | *"This scatter plot illustrates the 200 candidate pool in blue and the top 5 chosen candidates in red. The acquisition rule strikes a balance between exploitation (high predicted validation score) and exploration (high tree dispersion uncertainty)."* |
| **2:15 - 2:45** | **5. Benchmark Tab & Results** | Switch to the **📊 Validated Benchmark Results** tab. Show the test Macro-F1 comparison chart across the 4 unseen datasets (`madelon`, `diabetes`, `steel-plates-fault`, `ilpd`). | *"We evaluated MetaHPO-C against Random Search and standard Optuna TPE under an equalized 20-trial budget across 4 strictly held-out unseen datasets. On Madelon, MetaHPO-C achieved the highest test Macro-F1 ($0.8596$). On Steel-Plates-Fault, all methods achieved ceiling $1.0000$. Results are dataset-dependent."* |
| **2:45 - 3:00** | **6. Integrity & Limitations** | Switch to the **🔬 Methodology & Architecture** tab. Highlight the research integrity checklist. | *"We maintain scientific integrity: unseen datasets were never used for meta-training, test data was evaluated once, and we explicitly document that the uncertainty proxy is an empirical dispersion measure, not calibrated Bayesian probability."* |

---

# Part B: 5-Minute Technical Deep-Dive

Use this structured overview if the panel asks for a formal technical walkthrough:

### 1. The Cold-Start Problem in HPO
Standard HPO algorithms (such as Random Search or Bayesian Optimization with Gaussian Processes / TPE) start tabula rasa for every new dataset. When evaluation is computationally expensive, early iterations waste evaluation budget on suboptimal parameter regions.

### 2. Meta-Learning Dataset Construction
We executed 20 HPO trials across 7 diverse OpenML classification datasets (`adult`, `bank-marketing`, `credit-g`, `wdbc`, `spambase`, `vehicle`, `phoneme`), producing 140 supervised meta-learning instances. Each instance consists of:
$$\mathbf{z} = [\mathbf{m}_X, \boldsymbol{\theta}] \in \mathbb{R}^{30}$$
where $\mathbf{m}_X \in \mathbb{R}^{22}$ represents the canonical $\mathbf{X}$-only meta-features, $\boldsymbol{\theta} \in \mathbb{R}^8$ denotes the 8 XGBoost hyperparameters, and the target is the 3-fold Stratified CV Macro-F1 score $y \in [0, 1]$.

### 3. ExtraTrees Surrogate & Leave-One-Dataset-Out (LODO) Validation
We train an `ExtraTreesRegressor` ($T=300$ trees, `min_samples_leaf=2`, `max_features=1.0`). To rigorously evaluate generalization across task domains, we performed Leave-One-Dataset-Out (LODO) cross-validation across all 7 historical datasets (training on 6 datasets [120 trials] and evaluating on the 7th held-out dataset [20 trials]). The meta-model achieved a mean LODO MAE of $0.0573$ and a mean Spearman rank correlation of $\rho = +0.6720$.

### 4. Ensemble-Dispersion Uncertainty Proxy
For any candidate $\mathbf{x} = [\mathbf{m}_{\text{unseen}}, \boldsymbol{\theta}_k]$, individual tree predictions $p_1(\mathbf{x}), \dots, p_T(\mathbf{x})$ yield:
- **Surrogate Mean Prediction:** $\hat{\mu}(\mathbf{x}) = \frac{1}{T}\sum_{t=1}^T p_t(\mathbf{x})$
- **Ensemble Uncertainty Proxy:** $u(\mathbf{x}) = \sqrt{\frac{1}{T}\sum_{t=1}^T (p_t(\mathbf{x}) - \hat{\mu}(\mathbf{x}))^2}$
- **Relative Confidence:** $c(\mathbf{x}) = \frac{1}{1 + \frac{u(\mathbf{x})}{\text{median}(u) + \epsilon}} \in (0, 1]$

### 5. Confidence-Aware Acquisition & Warm-Start HPO
We sample $N=200$ configurations uniformly/log-uniformly across the 8-dimensional hyperparameter space and rank them via:
$$\alpha(\mathbf{x}) = \hat{\mu}(\mathbf{x}) + \beta \cdot u(\mathbf{x}), \quad \beta = 0.1$$
The top $K=5$ configurations are evaluated in the unseen dataset's 3-fold CV environment as warm-start trials (Trials 1–5), providing initial prior observations for 15 subsequent Optuna TPE trials (Trials 6–20).

---

# Part C: 25 Viva Defense Questions & Answers

### 1. What is Hyperparameter Optimization (HPO)?
**Answer:** HPO is the process of finding the configuration of external algorithm hyperparameters (e.g., learning rate, tree depth) that maximizes the validation performance of a learning model on a given dataset without manual trial-and-error.

### 2. What is the "cold-start" problem in HPO?
**Answer:** Standard HPO algorithms begin optimization with zero knowledge about the target dataset, starting the search from random points and wasting valuable computational budget exploring unpromising regions.

### 3. How does Meta-Learning solve the cold-start problem?
**Answer:** Meta-learning extracts dataset-level characteristics ("meta-features") and learns a surrogate performance response surface from prior HPO experiments across other datasets to recommend high-performing initial configurations for the new dataset.

### 4. What is a dataset meta-feature?
**Answer:** A meta-feature is a numerical descriptor summarizing the structural, statistical, or distributional properties of a dataset (such as sample count, feature ratios, mean skewness, or correlation).

### 5. Why are the meta-features in MetaHPO-C strictly $\mathbf{X}$-only?
**Answer:** Using target labels $y$ during meta-feature extraction on an unseen test dataset risks target leakage. Extracting descriptors purely from the input matrix $\mathbf{X}$ guarantees zero label leakage and allows warm-starting even on unlabeled or newly acquired data.

### 6. What are the 6 functional categories of the 22 X-only meta-features?
**Answer:**
1. **Dataset Size (5):** Sample count, feature count, ratios, and log transforms.
2. **Feature Composition (4):** Numeric/categorical counts and ratios.
3. **Distribution Statistics (7):** Means, standard deviations, min, max, and coefficient of variation.
4. **Missingness (2):** Overall missing cell ratio and ratio of features with missing values.
5. **Categorical Structure (2):** Mean and max categorical cardinality.
6. **Correlation (2):** Mean and max absolute pairwise feature correlations.

### 7. What machine learning model is tuned in this project?
**Answer:** Extreme Gradient Boosting (`XGBoost`), specifically `XGBClassifier` evaluated on multi-class and binary tabular classification tasks.

### 8. What are the 8 XGBoost hyperparameters in your search space?
**Answer:**
`n_estimators` ($[50, 300]$), `max_depth` ($[2, 8]$), `learning_rate` ($[0.01, 0.30]$ log), `subsample` ($[0.60, 1.00]$), `colsample_bytree` ($[0.60, 1.00]$), `min_child_weight` ($[1, 10]$), `gamma` ($[0.00, 2.00]$), and `reg_lambda` ($[0.10, 10.00]$ log). `reg_alpha` is not included.

### 9. What is the meta-model architecture?
**Answer:** An `ExtraTreesRegressor` (Extremely Randomized Trees) with 300 estimators, `min_samples_leaf=2`, `max_features=1.0`, and `random_state=42`.

### 10. Why use ExtraTrees instead of standard Random Forest or a Neural Network?
**Answer:** ExtraTrees introduces extreme randomization in split thresholds, which reduces variance on tabular meta-datasets, produces smoother response surfaces, executes fast parallel training, and directly provides tree-level ensemble disagreement without additional sub-model training.

### 11. Why predict performance $f(\mathbf{m}, \boldsymbol{\theta})$ instead of directly mapping $\mathbf{m} \to \boldsymbol{\theta}^*$?
**Answer:** Direct parameter regression assumes a single static optimal configuration exists per meta-feature vector, ignoring multi-modal optima. Learning a response-surface surrogate allows scoring arbitrary candidate pools and ranking multiple warm-start configurations.

### 12. What is Leave-One-Dataset-Out (LODO) Cross-Validation?
**Answer:** A rigorous meta-learning evaluation protocol where all $N=20$ trials of one historical dataset are held out, the meta-model is trained on the remaining 6 datasets ($120$ trials), and evaluated on the held-out dataset's trials. This is repeated for all 7 historical datasets.

### 13. What is the ensemble-dispersion uncertainty proxy?
**Answer:** It is the standard deviation across the individual decision tree predictions in the fitted ExtraTrees forest: $\sigma(x) = \text{std}(p_1(x), \dots, p_T(x))$. High dispersion indicates high model disagreement.

### 14. Is this uncertainty proxy a calibrated Bayesian probability?
**Answer:** **No.** It is an empirical ensemble-dispersion proxy reflecting tree disagreement. It is not a formal Bayesian posterior or calibrated probability distribution.

### 15. Why does tree disagreement correlate with prediction error?
**Answer:** When a query configuration lies in a sparsely sampled region of the meta-feature space or in a complex response valley, randomized decision trees make divergent split decisions, causing high ensemble variance which correlates empirically with higher generalization error ($r_s = +0.1799, p = 0.0334$).

### 16. What is the acquisition formula and what does $\beta=0.1$ represent?
**Answer:** The acquisition score is $\alpha = \hat{\mu} + \beta \cdot u$. The parameter $\beta=0.1$ controls the exploration-exploitation trade-off: $\hat{\mu}$ rewards configurations with high predicted performance, while $\beta \cdot u$ rewards configurations in under-explored parameter regions.

### 17. Why generate 200 candidate configurations?
**Answer:** $N=200$ uniformly and log-uniformly sampled configurations provide dense, unbiased coverage over the compact 8-dimensional search space while keeping meta-model inference time under $50$ milliseconds.

### 18. Why select exactly 5 warm-start configurations?
**Answer:** Under an overall 20-trial budget, 5 warm-start trials ($25\%$ of budget) inject strong domain priors while reserving 15 trials ($75\%$ of budget) for adaptive Optuna TPE search.

### 19. What optimization objective was used and why?
**Answer:** Macro-Averaged F1 score (`cv_macro_f1`). Macro-F1 treats all classes equally, penalizing models that ignore minority classes in imbalanced classification datasets.

### 20. How were the baseline algorithms configured?
**Answer:**
- **Random Search:** 20 pseudo-random trials sampled uniformly/log-uniformly ($\text{seed}=42$).
- **Optuna TPE:** 20 sequential trials using the Tree-structured Parzen Estimator ($\text{seed}=42$).
- Both operate on the exact same 3-fold Stratified CV splits on the 80% training set.

### 21. What were the 4 reserved unseen benchmark datasets?
**Answer:** `madelon` (OpenML 1485), `diabetes` (OpenML 37), `steel-plates-fault` (OpenML 1504), and `ilpd` (OpenML 1480). None of these were used during meta-model training.

### 22. What were the final held-out test Macro-F1 results on the 4 unseen datasets?
**Answer:**
- `madelon`: MetaHPO-C (**0.8596**) > Optuna (0.8538) > Random Search (0.8519).
- `diabetes`: Optuna (**0.7291**) > Random Search (0.7207) > MetaHPO-C (0.7068).
- `steel-plates-fault`: All three methods achieved **1.0000**.
- `ilpd`: Optuna (**0.5581**) > MetaHPO-C (0.5451) > Random Search (0.5357).

### 23. Does MetaHPO-C claim universal superiority over standard HPO?
**Answer:** **No.** The benchmark shows dataset-dependent outcomes. MetaHPO-C proved competitive and achieved the top score on `madelon`, but Optuna outperformed it on `diabetes` and `ilpd`. Paired statistical tests ($p > 0.05$) indicate feasibility rather than universal dominance.

### 24. What are the primary research limitations of this project?
**Answer:**
1. Small meta-training set (7 historical datasets, 140 trials).
2. Small unseen benchmark (4 datasets, single random seed).
3. Search space limited to 8 XGBoost hyperparameters.
4. Uncertainty is an uncalibrated dispersion heuristic.

### 25. What future improvements would you pursue with more time and resources?
**Answer:**
1. Scale historical experience to 50+ OpenML datasets and 100+ trials per dataset.
2. Incorporate Deep Kernel Learning or Gaussian Process surrogates for calibrated Bayesian posterior uncertainty.
3. Multi-fidelity evaluation (e.g., Hyperband / successive halving) for faster warm-start filtering.
4. Support multi-model algorithm selection across XGBoost, LightGBM, CatBoost, and Multi-Layer Perceptrons.
