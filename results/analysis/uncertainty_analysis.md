# Ensemble Dispersion Uncertainty Proxy Analysis

## Overall Summary across 140 LODO Trials

- **Overall Spearman Correlation ($r_s$):** 0.1799 (p=3.3384e-02)
- **Overall Pearson Correlation ($r$):** 0.0820 (p=3.3532e-01)
- **Low-Uncertainty MAE ($\le$ median 0.0879):** 0.1110
- **High-Uncertainty MAE (> median 0.0879):** 0.1160
- **MAE Gap (High - Low):** 0.0051

## Per-Dataset Correlation Breakdown

| dataset_name | n_trials | mean_uncertainty | median_uncertainty | mean_absolute_error | median_absolute_error | spearman_uncertainty_vs_error | pearson_uncertainty_vs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | 20 | 0.044106 | 0.043024 | 0.086071 | 0.080305 | 0.339850 | 0.517870 |
| bank-marketing | 20 | 0.093143 | 0.090247 | 0.027715 | 0.012436 | 0.320301 | 0.717355 |
| credit-g | 20 | 0.094777 | 0.087623 | 0.113473 | 0.101013 | 0.439098 | 0.940715 |
| phoneme | 20 | 0.093494 | 0.086179 | 0.076206 | 0.077762 | 0.111278 | -0.481018 |
| spambase | 20 | 0.128728 | 0.122007 | 0.114441 | 0.111456 | 0.717293 | 0.904426 |
| vehicle | 20 | 0.047426 | 0.043452 | 0.193045 | 0.190306 | 0.171429 | 0.237990 |
| wdbc | 20 | 0.123823 | 0.117875 | 0.183568 | 0.179256 | 0.715789 | 0.937463 |

> **Note:** The uncertainty measure is an **ensemble prediction dispersion proxy** (standard deviation across ExtraTrees estimators), not a statistically calibrated probability distribution.
