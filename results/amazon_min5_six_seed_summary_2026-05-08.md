# Amazon min5 Six-Seed Summary

Generated: 2026-05-08 02:48:57

NCL uses the safe Amazon min5 cluster count `num_clusters=36`, which matches the current FAISS points-per-centroid safety rule for this split.

## Aggregate

| Model                            | Seeds             |   Count | recall@10         | ndcg@10           | recall@50         | ndcg@50           |
|:---------------------------------|:------------------|--------:|:------------------|:------------------|:------------------|:------------------|
| MostPopIndependent               | 42/43/44/45/46/47 |       6 | 0.0282 +/- 0.0037 | 0.0133 +/- 0.0020 | 0.0851 +/- 0.0066 | 0.0256 +/- 0.0024 |
| Semantic-only                    | 42/43/44/45/46/47 |       6 | 0.0441 +/- 0.0062 | 0.0287 +/- 0.0053 | 0.0854 +/- 0.0058 | 0.0378 +/- 0.0051 |
| BPR                              | 42/43/44/45/46/47 |       6 | 0.0181 +/- 0.0042 | 0.0097 +/- 0.0024 | 0.0415 +/- 0.0064 | 0.0149 +/- 0.0027 |
| LightGCN                         | 42/43/44/45/46/47 |       6 | 0.0311 +/- 0.0036 | 0.0175 +/- 0.0025 | 0.0878 +/- 0.0026 | 0.0300 +/- 0.0019 |
| NCL                              | 42/43/44/45/46/47 |       6 | 0.0326 +/- 0.0052 | 0.0177 +/- 0.0028 | 0.0906 +/- 0.0041 | 0.0304 +/- 0.0022 |
| AsymLightGCN                     | 42/43/44/45/46/47 |       6 | 0.0465 +/- 0.0059 | 0.0232 +/- 0.0035 | 0.1297 +/- 0.0070 | 0.0413 +/- 0.0038 |
| AsymLightGCN (Gamma=0 eval-only) | 42/43/44/45/46/47 |       6 | 0.0460 +/- 0.0051 | 0.0232 +/- 0.0036 | 0.1285 +/- 0.0077 | 0.0411 +/- 0.0041 |

## Selected Rows

| Model                            |   Seed |   recall@10 |   ndcg@10 |   recall@50 |   ndcg@50 | Source CSV                                                                             |
|:---------------------------------|-------:|------------:|----------:|------------:|----------:|:---------------------------------------------------------------------------------------|
| AsymLightGCN                     |     42 |   0.0533    | 0.0261    |   0.1332    | 0.0431    | train_logs\final_full_graph_results_2026-05-04_23-39-32.csv                            |
| AsymLightGCN                     |     43 |   0.045     | 0.0217    |   0.1261    | 0.0393    | train_logs\final_full_graph_results_2026-05-05_20-11-53_partial.csv                    |
| AsymLightGCN                     |     44 |   0.0363    | 0.0168    |   0.1179    | 0.0346    | train_logs\final_full_graph_results_2026-05-05_20-13-34.csv                            |
| AsymLightGCN                     |     45 |   0.05      | 0.0245    |   0.1294    | 0.0417    | train_logs\final_full_graph_results_2026-05-08_02-37-32_partial.csv                    |
| AsymLightGCN                     |     46 |   0.045     | 0.0255    |   0.1346    | 0.045     | train_logs\final_full_graph_results_2026-05-08_02-38-21.csv                            |
| AsymLightGCN                     |     47 |   0.0494    | 0.0248    |   0.1372    | 0.0439    | train_logs\final_full_graph_results_2026-05-08_02-38-49.csv                            |
| AsymLightGCN (Gamma=0 eval-only) |     42 |   0.0512    | 0.0252    |   0.1316    | 0.0425    | train_logs\final_full_graph_results_2026-05-04_23-39-32.csv                            |
| AsymLightGCN (Gamma=0 eval-only) |     43 |   0.0455    | 0.0221    |   0.1216    | 0.0387    | train_logs\final_full_graph_results_2026-05-05_20-11-53_partial.csv                    |
| AsymLightGCN (Gamma=0 eval-only) |     44 |   0.0363    | 0.0168    |   0.1163    | 0.0342    | train_logs\final_full_graph_results_2026-05-05_20-13-34.csv                            |
| AsymLightGCN (Gamma=0 eval-only) |     45 |   0.0472    | 0.0233    |   0.1315    | 0.0416    | train_logs\final_full_graph_results_2026-05-08_02-37-32_partial.csv                    |
| AsymLightGCN (Gamma=0 eval-only) |     46 |   0.0487    | 0.0272    |   0.136     | 0.0459    | train_logs\final_full_graph_results_2026-05-08_02-38-21.csv                            |
| AsymLightGCN (Gamma=0 eval-only) |     47 |   0.0469    | 0.0246    |   0.1337    | 0.0437    | train_logs\final_full_graph_results_2026-05-08_02-38-49.csv                            |
| BPR                              |     42 |   0.0174    | 0.009     |   0.0371    | 0.0133    | train_logs/extra_baselines_results_2026-05-06_21-45-00.csv                             |
| BPR                              |     43 |   0.0181    | 0.0104    |   0.0394    | 0.0153    | train_logs/extra_baselines_results_2026-05-06_21-45-00.csv                             |
| BPR                              |     44 |   0.0186    | 0.0102    |   0.0405    | 0.0148    | train_logs/extra_baselines_results_2026-05-06_21-45-00.csv                             |
| BPR                              |     45 |   0.0215    | 0.0106    |   0.0517    | 0.0173    | train_logs/extra_baselines_results_2026-05-08_02-39-17.csv                             |
| BPR                              |     46 |   0.0105    | 0.0056    |   0.0341    | 0.0106    | train_logs/extra_baselines_results_2026-05-08_02-39-17.csv                             |
| BPR                              |     47 |   0.0226    | 0.0127    |   0.046     | 0.018     | train_logs/extra_baselines_results_2026-05-08_02-39-17.csv                             |
| LightGCN                         |     42 |   0.0307    | 0.0163    |   0.0911    | 0.0297    | train_logs\final_full_graph_results_2026-05-04_23-39-32.csv                            |
| LightGCN                         |     43 |   0.0258    | 0.0132    |   0.0877    | 0.0268    | train_logs\final_full_graph_results_2026-05-05_20-11-53_partial.csv                    |
| LightGCN                         |     44 |   0.0302    | 0.018     |   0.0834    | 0.0296    | train_logs\final_full_graph_results_2026-05-05_20-13-34.csv                            |
| LightGCN                         |     45 |   0.0308    | 0.0185    |   0.089     | 0.0313    | train_logs\final_full_graph_results_2026-05-08_02-37-32_partial.csv                    |
| LightGCN                         |     46 |   0.0369    | 0.0187    |   0.089     | 0.0303    | train_logs\final_full_graph_results_2026-05-08_02-38-21.csv                            |
| LightGCN                         |     47 |   0.0324    | 0.0204    |   0.0865    | 0.0325    | train_logs\final_full_graph_results_2026-05-08_02-38-49.csv                            |
| MostPopIndependent               |     42 |   0.0274482 | 0.0132387 |   0.0894394 | 0.0267841 | train_logs/mostpop_amazon_min_degree_6seed_2026-05-08.csv                              |
| MostPopIndependent               |     43 |   0.0257037 | 0.0118382 |   0.0795534 | 0.0233641 | train_logs/mostpop_amazon_min_degree_6seed_2026-05-08.csv                              |
| MostPopIndependent               |     44 |   0.0230286 | 0.0106754 |   0.0755408 | 0.0219926 | train_logs/mostpop_amazon_min_degree_6seed_2026-05-08.csv                              |
| MostPopIndependent               |     45 |   0.0285947 | 0.0131152 |   0.0931076 | 0.0269927 | train_logs/mostpop_amazon_min_degree_6seed_2026-05-08.csv                              |
| MostPopIndependent               |     46 |   0.031519  | 0.0164444 |   0.0842056 | 0.0279496 | train_logs/mostpop_amazon_min_degree_6seed_2026-05-08.csv                              |
| MostPopIndependent               |     47 |   0.0330309 | 0.0145031 |   0.0888579 | 0.026475  | train_logs/mostpop_amazon_min_degree_6seed_2026-05-08.csv                              |
| NCL                              |     42 |   0.033     | 0.0174    |   0.0908    | 0.0303    | train_logs\final_full_graph_results_2026-05-08_02-44-26.csv                            |
| NCL                              |     43 |   0.0274    | 0.0149    |   0.0935    | 0.0293    | train_logs\final_full_graph_results_2026-05-08_02-45-08.csv                            |
| NCL                              |     44 |   0.0268    | 0.0139    |   0.0882    | 0.0271    | train_logs\final_full_graph_results_2026-05-08_02-45-49.csv                            |
| NCL                              |     45 |   0.0363    | 0.0194    |   0.0882    | 0.0308    | train_logs\final_full_graph_results_2026-05-08_02-46-31.csv                            |
| NCL                              |     46 |   0.0402    | 0.0212    |   0.097     | 0.0336    | train_logs\final_full_graph_results_2026-05-08_02-47-13.csv                            |
| NCL                              |     47 |   0.0316    | 0.0194    |   0.0859    | 0.0313    | train_logs\final_full_graph_results_2026-05-08_02-47-54.csv                            |
| Semantic-only                    |     42 |   0.042568  | 0.0238589 |   0.0841105 | 0.0330521 | train_logs\semantic_only_amazon_reduced_min5_fixed_seed42_2026-05-04_semantic_only.csv |
| Semantic-only                    |     43 |   0.0391952 | 0.029848  |   0.0801931 | 0.0387177 | train_logs\semantic_only_amazon_reduced_min5_fixed_seed43_2026-05-04_semantic_only.csv |
| Semantic-only                    |     44 |   0.0393115 | 0.0229122 |   0.0832752 | 0.032231  | train_logs\semantic_only_amazon_reduced_min5_fixed_seed44_2026-05-04_semantic_only.csv |
| Semantic-only                    |     45 |   0.0452431 | 0.0321924 |   0.0825026 | 0.0406226 | train_logs\semantic_only_amazon_min5_seed45_2026-05-08_semantic_only.csv               |
| Semantic-only                    |     46 |   0.0422191 | 0.0267655 |   0.0857176 | 0.0362475 | train_logs\semantic_only_amazon_min5_seed46_2026-05-08_semantic_only.csv               |
| Semantic-only                    |     47 |   0.0558269 | 0.036835  |   0.0967501 | 0.0457434 | train_logs\semantic_only_amazon_min5_seed47_2026-05-08_semantic_only.csv               |

## Missing Rows

None.
