# MetaHPO-C: Complete Results Tables & Empirical Data

This document provides complete, verified tabular results across all phases of the **MetaHPO-C** project. All figures and values are extracted directly from authoritative project artifacts in `results/experiments/` and `results/analysis/`.

---

## Table 1: Historical Meta-Dataset Summary (Phase 5)

Data Source: `results/experiments/historical/historical_dataset_summary.csv`

| Dataset Name | Total Samples ($N$) | Features ($D$) | Descriptive Classes | Best 3-Fold CV Macro-F1 | Best Trial Index | Total Trials | Mean Coeff. of Variation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `adult` | 48,842 | 14 | 2 | 0.792749 | 11 | 20 | 0.405946 |
| `bank-marketing` | 45,211 | 16 | 2 | 0.751809 | 0 | 20 | 1.657064 |
| `credit-g` | 1,000 | 20 | 2 | 0.730248 | 17 | 20 | 0.463782 |
| `wdbc` | 569 | 30 | 2 | 0.962443 | 19 | 20 | 0.486451 |
| `spambase` | 4,601 | 57 | 2 | 0.952936 | 13 | 20 | 5.188774 |
| `vehicle` | 846 | 18 | 4 | 0.778841 | 17 | 20 | 0.238795 |
| `phoneme` | 5,404 | 5 | 2 | 0.887258 | 15 | 20 | 1000.000000 |

---

## Table 2: Leave-One-Dataset-Out (LODO) Meta-Model Validation (Phase 7)

Data Source: `results/experiments/meta_learning/lodo_metrics.json`  
Model: `ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=1.0, random_state=42)`

| Held-Out Dataset | LODO MAE | LODO RMSE | LODO $R^2$ | Spearman Rank Corr ($r_s$) | Baseline Mean Pred MAE | Baseline Mean Pred $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `adult` | 0.086071 | 0.088833 | -1.651470 | **+0.663158** | 0.032674 | -0.358717 |
| `bank-marketing` | 0.027715 | 0.044147 | +0.684042 | **+0.624060** | 0.101292 | -1.663331 |
| `credit-g` | 0.113473 | 0.119101 | -1.751686 | **+0.796992** | 0.166703 | -5.390810 |
| `phoneme` | 0.076206 | 0.078835 | -0.577838 | **+0.723308** | 0.066891 | -0.388481 |
| `spambase` | 0.114441 | 0.114858 | -92.993277 | **+0.855639** | 0.166412 | -197.309193 |
| `vehicle` | 0.193045 | 0.193758 | -64.851733 | **+0.500752** | 0.079983 | -11.221178 |
| `wdbc` | 0.183568 | 0.184427 | -235.494322 | **+0.694737** | 0.175122 | -213.234169 |
| **Overall Summary** | **0.113503** | **0.128167** | **-0.191166** | **Pooled: +0.150362** | **0.112725** | **-0.289689** |
| **Mean Within-Dataset** | — | — | — | **Mean: +0.694092** | — | — |

---

## Table 3: LODO Uncertainty Proxy vs. Prediction Error Breakdown (Phase 8 & 11)

Data Source: `results/experiments/meta_learning/uncertainty_metrics.json` & `results/analysis/uncertainty_analysis.csv`

| Dataset | Sample Count ($N$) | Mean Uncertainty Proxy ($\sigma$) | Median Uncertainty Proxy | Mean Absolute Error (MAE) | Spearman Correlation ($r_s$) | Pearson Correlation ($r$) | Low-Uncertainty MAE | High-Uncertainty MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `adult` | 20 | 0.044106 | 0.043024 | 0.086071 | +0.339850 | +0.517870 | 0.078552 | 0.093589 |
| `bank-marketing` | 20 | 0.093143 | 0.090247 | 0.027715 | +0.320301 | +0.717355 | 0.013584 | 0.041845 |
| `credit-g` | 20 | 0.094777 | 0.087623 | 0.113473 | +0.439098 | +0.940715 | 0.097926 | 0.129020 |
| `phoneme` | 20 | 0.093494 | 0.086179 | 0.076206 | +0.111278 | -0.481018 | 0.077709 | 0.074703 |
| `spambase` | 20 | 0.128728 | 0.122007 | 0.114441 | +0.717293 | +0.904426 | 0.109858 | 0.119025 |
| `vehicle` | 20 | 0.047426 | 0.043452 | 0.193045 | +0.171429 | +0.237990 | 0.194198 | 0.191892 |
| `wdbc` | 20 | 0.123823 | 0.117875 | 0.183568 | +0.715789 | +0.937463 | 0.172838 | 0.194298 |
| **Overall Pooled** | **140** | **0.089357** | **0.087879** | **0.113503** | **+0.179944 ($p=0.0334$)** | **+0.082025** | **0.110971** | **0.116035** |

---

## Table 4: Reserved Unseen Benchmark Comprehensive Results (Phase 10 & 11)

Data Source: `results/analysis/final_results_table.csv`

| Dataset Name | Method | Best 3-Fold CV Macro-F1 | Best Trial Index | Held-Out Test Macro-F1 | Held-Out Test Accuracy | HPO Runtime (s) | Total Runtime (s) | Overhead (s) | Total Trials |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `diabetes` | `metahpo_c` | 0.739259 | 7 | 0.706834 | 0.740260 | 5.635990 | 5.942170 | 0.306180 | 20 |
| `diabetes` | `optuna_tpe` | **0.757512** | 15 | **0.729074** | **0.753247** | 6.498883 | 6.775762 | 0.276879 | 20 |
| `diabetes` | `random_search` | 0.753221 | 18 | 0.720742 | 0.746753 | 6.559876 | 6.564407 | 0.004531 | 20 |
| `ilpd` | `metahpo_c` | 0.631192 | 19 | 0.545062 | 0.615385 | 3.572625 | 3.759816 | 0.187191 | 20 |
| `ilpd` | `optuna_tpe` | 0.618532 | 18 | **0.558063** | **0.666667** | 3.127639 | 3.252348 | 0.124709 | 20 |
| `ilpd` | `random_search` | **0.633479** | 11 | 0.535714 | 0.623932 | 3.029203 | 3.032591 | 0.003388 | 20 |
| `madelon` | `metahpo_c` | 0.804724 | 9 | **0.859590** | **0.859615** | 182.467790 | 183.613346 | 1.145556 | 20 |
| `madelon` | `optuna_tpe` | **0.811022** | 12 | 0.853844 | 0.853846 | 159.302500 | 159.445204 | 0.142704 | 20 |
| `madelon` | `random_search` | 0.802351 | 8 | 0.851918 | 0.851923 | 135.864014 | 135.875878 | 0.011864 | 20 |
| `steel-plates-fault` | `metahpo_c` | **1.000000** | 2 | **1.000000** | **1.000000** | 7.290968 | 7.474795 | 0.183827 | 20 |
| `steel-plates-fault` | `optuna_tpe` | **1.000000** | 4 | **1.000000** | **1.000000** | 7.462924 | 7.592041 | 0.129117 | 20 |
| `steel-plates-fault` | `random_search` | **1.000000** | 0 | **1.000000** | **1.000000** | 6.091925 | 6.095417 | 0.003492 | 20 |

---

## Table 5: Per-Dataset Differences (MetaHPO-C vs. Baselines)

Data Source: `results/analysis/per_dataset_comparison.csv`

| Dataset | Baseline Method | $\Delta$ CV Macro-F1 (Abs) | $\Delta$ CV Macro-F1 (%) | $\Delta$ Test Macro-F1 (Abs) | $\Delta$ Test Macro-F1 (%) | $\Delta$ Test Accuracy (Abs) | $\Delta$ Test Accuracy (%) | Runtime Diff (s) | Runtime Overhead (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `madelon` | `random_search` | +0.002373 | +0.30% | +0.007672 | +0.90% | +0.007692 | +0.90% | +47.737468 | +35.13% |
| `madelon` | `optuna_tpe` | -0.006298 | -0.78% | +0.005746 | +0.67% | +0.005769 | +0.68% | +24.168142 | +15.16% |
| `diabetes` | `random_search` | -0.013962 | -1.85% | -0.013908 | -1.93% | -0.006494 | -0.87% | -0.622237 | -9.48% |
| `diabetes` | `optuna_tpe` | -0.018253 | -2.41% | -0.022240 | -3.05% | -0.012987 | -1.72% | -0.833592 | -12.30% |
| `steel-plates-fault` | `random_search` | 0.000000 | 0.00% | 0.000000 | 0.00% | 0.000000 | 0.00% | +1.379378 | +22.63% |
| `steel-plates-fault` | `optuna_tpe` | 0.000000 | 0.00% | 0.000000 | 0.00% | 0.000000 | 0.00% | -0.117247 | -1.54% |
| `ilpd` | `random_search` | -0.002287 | -0.36% | +0.009347 | +1.74% | -0.008547 | -1.37% | +0.727225 | +23.98% |
| `ilpd` | `optuna_tpe` | +0.012660 | +2.05% | -0.013001 | -2.33% | -0.051282 | -7.69% | +0.507468 | +15.60% |

---

## Table 6: Optimization Trajectory Checkpoints (Best-so-Far CV Macro-F1)

Data Source: `results/analysis/early_search_comparison.csv`

| Dataset Name | Trial Checkpoint | MetaHPO-C Score | Random Search Score | Optuna/TPE Score | $\Delta$ vs. Random (Abs) | $\Delta$ vs. Random (%) | $\Delta$ vs. Optuna (Abs) | $\Delta$ vs. Optuna (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `madelon` | Trial 1 | 0.765184 | 0.791781 | 0.761854 | -0.026597 | -3.36% | +0.003330 | +0.44% |
| `madelon` | Trial 5 | 0.796128 | 0.792128 | 0.804724 | +0.004000 | +0.50% | -0.008596 | -1.07% |
| `madelon` | Trial 10 | 0.804724 | 0.802351 | 0.804724 | +0.002373 | +0.30% | 0.000000 | 0.00% |
| `madelon` | Trial 20 | 0.804724 | 0.802351 | 0.811022 | +0.002373 | +0.30% | -0.006298 | -0.78% |
| `diabetes` | Trial 1 | 0.710777 | 0.733814 | 0.723899 | -0.023037 | -3.14% | -0.013122 | -1.81% |
| `diabetes` | Trial 5 | 0.728035 | 0.744453 | 0.739259 | -0.016418 | -2.21% | -0.011224 | -1.52% |
| `diabetes` | Trial 10 | 0.739259 | 0.744453 | 0.739820 | -0.005194 | -0.70% | -0.000561 | -0.08% |
| `diabetes` | Trial 20 | 0.739259 | 0.753221 | 0.757512 | -0.013962 | -1.85% | -0.018253 | -2.41% |
| `steel-plates-fault` | Trial 1 | 0.998580 | 1.000000 | 0.997867 | -0.001420 | -0.14% | +0.000713 | +0.07% |
| `steel-plates-fault` | Trial 5 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.00% | 0.000000 | 0.00% |
| `steel-plates-fault` | Trial 10 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.00% | 0.000000 | 0.00% |
| `steel-plates-fault` | Trial 20 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.00% | 0.000000 | 0.00% |
| `ilpd` | Trial 1 | 0.597337 | 0.578883 | 0.598978 | +0.018454 | +3.19% | -0.001641 | -0.27% |
| `ilpd` | Trial 5 | 0.624457 | 0.620250 | 0.611752 | +0.004207 | +0.68% | +0.012705 | +2.08% |
| `ilpd` | Trial 10 | 0.624457 | 0.620250 | 0.618120 | +0.004207 | +0.68% | +0.006337 | +1.03% |
| `ilpd` | Trial 20 | 0.631192 | 0.633479 | 0.618532 | -0.002287 | -0.36% | +0.012660 | +2.05% |

---

## Table 7: Warm-Start Diagnostic Summary (Phase 11)

Data Source: `results/analysis/warm_start_analysis.csv`

| Dataset Name | Warm-Start Best CV F1 | Warm-Start Best Trial | Post-WS Best CV F1 | Final Best CV F1 | TPE Exploration Gain | Best Found in Warm Start | Trial 5 Score (MetaHPO-C) | Trial 5 Score (Random) | Trial 5 Score (Optuna) | Trial 5 $\Delta$ vs. Random | Trial 5 $\Delta$ vs. Optuna |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `madelon` | 0.796128 | 4 | 0.804724 | 0.804724 | 0.008596 | False | 0.796128 | 0.792128 | 0.804724 | +0.004000 | -0.008596 |
| `diabetes` | 0.728035 | 4 | 0.739259 | 0.739259 | 0.011224 | False | 0.728035 | 0.744453 | 0.739259 | -0.016418 | -0.011224 |
| `steel-plates-fault` | 1.000000 | 3 | 1.000000 | 1.000000 | 0.000000 | **True** | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| `ilpd` | 0.624457 | 2 | 0.631192 | 0.631192 | 0.006735 | False | 0.624457 | 0.620250 | 0.611752 | +0.004207 | +0.012705 |

---

## Table 8: Paired Statistical Hypothesis Testing (Phase 11)

Data Source: `results/analysis/statistical_tests.csv`  
Sample Size: $n=4$ Unseen Datasets (`madelon`, `diabetes`, `steel-plates-fault`, `ilpd`), Seed = 42

| Comparison | Target Metric | Paired Diff (`madelon`) | Paired Diff (`diabetes`) | Paired Diff (`steel`) | Paired Diff (`ilpd`) | Mean Paired Diff ($\overline{\Delta}$) | Paired $t$-Statistic | Paired $t$ $p$-Value | Wilcoxon Statistic | Wilcoxon $p$-Value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `metahpo_c_vs_random_search` | Best CV Macro-F1 | +0.002373 | -0.013962 | 0.000000 | -0.002287 | -0.003469 | -0.957098 | 0.409125 | 3.0 | 0.7500 |
| `metahpo_c_vs_random_search` | Test Macro-F1 | +0.007672 | -0.013908 | 0.000000 | +0.009347 | +0.000778 | +0.146729 | 0.892652 | 4.0 | 1.0000 |
| `metahpo_c_vs_random_search` | Test Accuracy | +0.007692 | -0.006494 | 0.000000 | -0.008547 | -0.001837 | -0.501705 | 0.650381 | 3.0 | 0.7500 |
| `metahpo_c_vs_random_search` | Total Runtime (s) | +47.737468 | -0.622237 | +1.379378 | +0.727225 | +12.305458 | +1.041245 | 0.374297 | 1.0 | 0.2500 |
| `metahpo_c_vs_optuna_tpe` | Best CV Macro-F1 | -0.006298 | -0.018253 | 0.000000 | +0.012660 | -0.002973 | -0.461579 | 0.675794 | 3.0 | 0.7500 |
| `metahpo_c_vs_optuna_tpe` | Test Macro-F1 | +0.005746 | -0.022240 | 0.000000 | -0.013001 | -0.007374 | -1.166908 | 0.327568 | 2.0 | 0.5000 |
| `metahpo_c_vs_optuna_tpe` | Test Accuracy | +0.005769 | -0.012987 | 0.000000 | -0.051282 | -0.014625 | -1.139635 | 0.337192 | 2.0 | 0.5000 |
| `metahpo_c_vs_optuna_tpe` | Total Runtime (s) | +24.168142 | -0.833592 | -0.117247 | +0.507468 | +5.931193 | +0.974699 | 0.401597 | 4.0 | 0.8750 |
