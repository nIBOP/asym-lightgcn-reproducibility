# User-Level Significance Analysis

Per-user Recall@K and NDCG@K were recomputed with full-sort candidate masking for the selected Amazon min5 test split. Deltas are averaged per user across available seeds before bootstrap and non-parametric tests.

Bootstrap samples: 10000

## Aggregate Check

| Dataset                   |   Seed | Model              |   Users |   recall@10 |   ndcg@10 |   recall@50 |   ndcg@50 |
|:--------------------------|-------:|:-------------------|--------:|------------:|----------:|------------:|----------:|
| clean_amazon_reduced_min5 |     42 | AsymLightGCN       |    1433 |    0.053291 |  0.026066 |    0.133178 |  0.043117 |
| clean_amazon_reduced_min5 |     42 | LightGCN           |    1433 |    0.030705 |  0.016349 |    0.091068 |  0.029739 |
| clean_amazon_reduced_min5 |     42 | MostPopIndependent |    1433 |    0.027448 |  0.013239 |    0.089439 |  0.026784 |
| clean_amazon_reduced_min5 |     42 | NCL                |    1433 |    0.033031 |  0.017418 |    0.090777 |  0.030270 |
| clean_amazon_reduced_min5 |     42 | Semantic-only      |    1433 |    0.042568 |  0.023859 |    0.084110 |  0.033052 |
| clean_amazon_reduced_min5 |     43 | AsymLightGCN       |    1433 |    0.045045 |  0.021711 |    0.126136 |  0.039304 |
| clean_amazon_reduced_min5 |     43 | LightGCN           |    1433 |    0.025820 |  0.013186 |    0.087695 |  0.026796 |
| clean_amazon_reduced_min5 |     43 | MostPopIndependent |    1433 |    0.025704 |  0.011838 |    0.079553 |  0.023364 |
| clean_amazon_reduced_min5 |     43 | NCL                |    1433 |    0.027448 |  0.014920 |    0.093510 |  0.029250 |
| clean_amazon_reduced_min5 |     43 | Semantic-only      |    1433 |    0.039195 |  0.029848 |    0.080193 |  0.038718 |
| clean_amazon_reduced_min5 |     44 | AsymLightGCN       |    1433 |    0.036251 |  0.016826 |    0.117942 |  0.034577 |
| clean_amazon_reduced_min5 |     44 | LightGCN           |    1433 |    0.030240 |  0.017989 |    0.083450 |  0.029599 |
| clean_amazon_reduced_min5 |     44 | MostPopIndependent |    1433 |    0.023029 |  0.010675 |    0.075541 |  0.021993 |
| clean_amazon_reduced_min5 |     44 | NCL                |    1433 |    0.026750 |  0.013939 |    0.088218 |  0.027059 |
| clean_amazon_reduced_min5 |     44 | Semantic-only      |    1433 |    0.039311 |  0.022912 |    0.083275 |  0.032231 |
| clean_amazon_reduced_min5 |     45 | AsymLightGCN       |    1433 |    0.049995 |  0.024524 |    0.129430 |  0.041710 |
| clean_amazon_reduced_min5 |     45 | LightGCN           |    1433 |    0.030805 |  0.018497 |    0.088958 |  0.031313 |
| clean_amazon_reduced_min5 |     45 | MostPopIndependent |    1433 |    0.028595 |  0.013115 |    0.093108 |  0.026993 |
| clean_amazon_reduced_min5 |     45 | NCL                |    1433 |    0.036288 |  0.019381 |    0.088202 |  0.030753 |
| clean_amazon_reduced_min5 |     45 | Semantic-only      |    1433 |    0.045243 |  0.032192 |    0.082503 |  0.040623 |
| clean_amazon_reduced_min5 |     46 | AsymLightGCN       |    1433 |    0.045032 |  0.025464 |    0.134550 |  0.044964 |
| clean_amazon_reduced_min5 |     46 | LightGCN           |    1433 |    0.036869 |  0.018651 |    0.088974 |  0.030270 |
| clean_amazon_reduced_min5 |     46 | MostPopIndependent |    1433 |    0.031519 |  0.016444 |    0.084206 |  0.027950 |
| clean_amazon_reduced_min5 |     46 | NCL                |    1433 |    0.040242 |  0.021227 |    0.097023 |  0.033565 |
| clean_amazon_reduced_min5 |     46 | Semantic-only      |    1433 |    0.042219 |  0.026765 |    0.085718 |  0.036248 |
| clean_amazon_reduced_min5 |     47 | AsymLightGCN       |    1433 |    0.049430 |  0.024817 |    0.137160 |  0.043898 |
| clean_amazon_reduced_min5 |     47 | LightGCN           |    1433 |    0.032449 |  0.020448 |    0.086450 |  0.032529 |
| clean_amazon_reduced_min5 |     47 | MostPopIndependent |    1433 |    0.033031 |  0.014503 |    0.088858 |  0.026475 |
| clean_amazon_reduced_min5 |     47 | NCL                |    1433 |    0.031635 |  0.019441 |    0.085892 |  0.031315 |
| clean_amazon_reduced_min5 |     47 | Semantic-only      |    1433 |    0.055827 |  0.036835 |    0.096750 |  0.045743 |

## Paired User-Level Tests

| Dataset                   | Comparison                        | Metric    |   Paired Seed-User Rows |   Users |   Mean Delta |   Median Delta |   Bootstrap 95% CI Low |   Bootstrap 95% CI High | Wilcoxon p   | Sign p   |   Positive Users |   Negative Users |   Zero Users |
|:--------------------------|:----------------------------------|:----------|------------------------:|--------:|-------------:|---------------:|-----------------------:|------------------------:|:-------------|:---------|-----------------:|-----------------:|-------------:|
| clean_amazon_reduced_min5 | AsymLightGCN - LightGCN           | recall@10 |                    8598 |    1433 |     0.015359 |              0 |               0.00958  |                0.021109 | <0.0001      | <0.0001  |              211 |              100 |         1122 |
| clean_amazon_reduced_min5 | AsymLightGCN - LightGCN           | ndcg@10   |                    8598 |    1433 |     0.005715 |              0 |               0.00217  |                0.009277 | <0.0001      | <0.0001  |              223 |              140 |         1070 |
| clean_amazon_reduced_min5 | AsymLightGCN - LightGCN           | recall@50 |                    8598 |    1433 |     0.041967 |              0 |               0.032887 |                0.051252 | <0.0001      | <0.0001  |              392 |              212 |          829 |
| clean_amazon_reduced_min5 | AsymLightGCN - LightGCN           | ndcg@50   |                    8598 |    1433 |     0.011221 |              0 |               0.007368 |                0.01513  | <0.0001      | <0.0001  |              451 |              303 |          679 |
| clean_amazon_reduced_min5 | AsymLightGCN - NCL                | recall@10 |                    8598 |    1433 |     0.013942 |              0 |               0.00802  |                0.019954 | <0.0001      | <0.0001  |              201 |               99 |         1133 |
| clean_amazon_reduced_min5 | AsymLightGCN - NCL                | ndcg@10   |                    8598 |    1433 |     0.005514 |              0 |               0.002059 |                0.009026 | 0.0009       | <0.0001  |              218 |              136 |         1079 |
| clean_amazon_reduced_min5 | AsymLightGCN - NCL                | recall@50 |                    8598 |    1433 |     0.039129 |              0 |               0.029992 |                0.048365 | <0.0001      | <0.0001  |              376 |              204 |          853 |
| clean_amazon_reduced_min5 | AsymLightGCN - NCL                | ndcg@50   |                    8598 |    1433 |     0.010893 |              0 |               0.007256 |                0.014582 | <0.0001      | <0.0001  |              441 |              297 |          695 |
| clean_amazon_reduced_min5 | AsymLightGCN - Semantic-only      | recall@10 |                    8598 |    1433 |     0.002447 |              0 |              -0.005662 |                0.010075 | 0.1455       | <0.0001  |              233 |              152 |         1048 |
| clean_amazon_reduced_min5 | AsymLightGCN - Semantic-only      | ndcg@10   |                    8598 |    1433 |    -0.005501 |              0 |              -0.011103 |               -6.2e-05  | 0.4339       | 0.0010   |              241 |              173 |         1019 |
| clean_amazon_reduced_min5 | AsymLightGCN - Semantic-only      | recall@50 |                    8598 |    1433 |     0.044308 |              0 |               0.032798 |                0.05591  | <0.0001      | <0.0001  |              461 |              224 |          748 |
| clean_amazon_reduced_min5 | AsymLightGCN - Semantic-only      | ndcg@50   |                    8598 |    1433 |     0.003493 |              0 |              -0.002187 |                0.009084 | <0.0001      | <0.0001  |              493 |              284 |          656 |
| clean_amazon_reduced_min5 | AsymLightGCN - MostPopIndependent | recall@10 |                    8598 |    1433 |     0.018286 |              0 |               0.013261 |                0.023503 | <0.0001      | <0.0001  |              180 |               61 |         1192 |
| clean_amazon_reduced_min5 | AsymLightGCN - MostPopIndependent | ndcg@10   |                    8598 |    1433 |     0.009932 |              0 |               0.00732  |                0.012711 | <0.0001      | <0.0001  |              225 |               89 |         1119 |
| clean_amazon_reduced_min5 | AsymLightGCN - MostPopIndependent | recall@50 |                    8598 |    1433 |     0.044615 |              0 |               0.0363   |                0.053059 | <0.0001      | <0.0001  |              335 |              123 |          975 |
| clean_amazon_reduced_min5 | AsymLightGCN - MostPopIndependent | ndcg@50   |                    8598 |    1433 |     0.015669 |              0 |               0.012508 |                0.018878 | <0.0001      | <0.0001  |              468 |              211 |          754 |
