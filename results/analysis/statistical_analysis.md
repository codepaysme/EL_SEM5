# Statistical Analysis & Hypothesis Testing (Phase 10 Unseen Datasets)

## Methodological Notes on Statistical Power

> **IMPORTANT METHODOLOGICAL DISCLAIMER:**
> 1. The independent experimental units in this benchmark are the **four reserved unseen datasets** ($n=4$), evaluated with a single fixed random seed (42).
> 2. Treating the 240 individual HPO trials as independent samples would commit a pseudoreplication error; thus statistical testing is performed strictly across datasets.
> 3. Because $n=4$, statistical power is severely limited. P-values are reported for transparency, but non-significant p-values **must not** be interpreted as evidence of method equivalence, nor do significant differences establish universal generalizability.

## Paired Comparison Test Results

| comparison | metric | n_datasets | paired_diff_d1_madelon | paired_diff_d2_diabetes | paired_diff_d3_steel | paired_diff_d4_ilpd | mean_paired_diff | paired_t_stat | paired_t_pval | wilcoxon_stat | wilcoxon_pval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metahpo_c_vs_random_search | best_cv_macro_f1 | 4 | 0.002373 | -0.013962 | 0.00 | -0.002287 | -0.003469 | -0.957098 | 0.409125 | 3.000000 | 0.750000 |
| metahpo_c_vs_random_search | test_macro_f1 | 4 | 0.007672 | -0.013908 | 0.00 | 0.009347 | 0.000778 | 0.146729 | 0.892652 | 4.000000 | 1.000000 |
| metahpo_c_vs_random_search | test_accuracy | 4 | 0.007692 | -0.006494 | 0.00 | -0.008547 | -0.001837 | -0.501705 | 0.650381 | 3.000000 | 0.750000 |
| metahpo_c_vs_random_search | total_runtime_seconds | 4 | 47.737468 | -0.622237 | 1.379378 | 0.727225 | 12.305458 | 1.041245 | 0.374297 | 1.000000 | 0.250000 |
| metahpo_c_vs_optuna_tpe | best_cv_macro_f1 | 4 | -0.006298 | -0.018253 | 0.00 | 0.012660 | -0.002973 | -0.461579 | 0.675794 | 3.000000 | 0.750000 |
| metahpo_c_vs_optuna_tpe | test_macro_f1 | 4 | 0.005746 | -0.022240 | 0.00 | -0.013001 | -0.007374 | -1.166908 | 0.327568 | 2.000000 | 0.500000 |
| metahpo_c_vs_optuna_tpe | test_accuracy | 4 | 0.005769 | -0.012987 | 0.00 | -0.051282 | -0.014625 | -1.139635 | 0.337192 | 2.000000 | 0.500000 |
| metahpo_c_vs_optuna_tpe | total_runtime_seconds | 4 | 24.168142 | -0.833592 | -0.117247 | 0.507468 | 5.931193 | 0.974699 | 0.401597 | 4.000000 | 0.875000 |
