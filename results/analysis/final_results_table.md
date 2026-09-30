# Final Benchmark Results Table

| dataset_name | method | best_cv_macro_f1 | best_trial_number | test_macro_f1 | test_accuracy | hpo_runtime_seconds | total_runtime_seconds | overhead_seconds | n_trials |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diabetes | metahpo_c | 0.739259 | 8 | 0.706834 | 0.740260 | 5.635990 | 5.942170 | 0.306180 | 20 |
| diabetes | optuna_tpe | 0.757512 | 15 | 0.729074 | 0.753247 | 6.498883 | 6.775762 | 0.276879 | 20 |
| diabetes | random_search | 0.753221 | 16 | 0.720742 | 0.746753 | 6.559876 | 6.564407 | 0.004531 | 20 |
| ilpd | metahpo_c | 0.631192 | 19 | 0.545062 | 0.615385 | 3.572625 | 3.759816 | 0.187191 | 20 |
| ilpd | optuna_tpe | 0.618532 | 10 | 0.558063 | 0.666667 | 3.127639 | 3.252348 | 0.124709 | 20 |
| ilpd | random_search | 0.633479 | 18 | 0.535714 | 0.623932 | 3.029203 | 3.032591 | 0.003389 | 20 |
| madelon | metahpo_c | 0.804724 | 6 | 0.859590 | 0.859615 | 182.467790 | 183.613346 | 1.145556 | 20 |
| madelon | optuna_tpe | 0.811022 | 14 | 0.853844 | 0.853846 | 159.302500 | 159.445204 | 0.142704 | 20 |
| madelon | random_search | 0.802351 | 7 | 0.851918 | 0.851923 | 135.864014 | 135.875878 | 0.011864 | 20 |
| steel-plates-fault | metahpo_c | 1.000000 | 2 | 1.000000 | 1.000000 | 7.290968 | 7.474795 | 0.183826 | 20 |
| steel-plates-fault | optuna_tpe | 1.000000 | 1 | 1.000000 | 1.000000 | 7.462924 | 7.592041 | 0.129117 | 20 |
| steel-plates-fault | random_search | 1.000000 | 0 | 1.000000 | 1.000000 | 6.091925 | 6.095417 | 0.003492 | 20 |
