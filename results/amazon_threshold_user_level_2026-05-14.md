# Amazon Threshold User-Level Significance

Per-user Recall@K and NDCG@K were recomputed with full-sort candidate masking for Amazon min4/min6/min8/min10. Deltas are averaged per user across available seeds before bootstrap and paired non-parametric tests.

Bootstrap samples: 5000

## Aggregate Check

| Dataset                    |   Seed | Model              |   Users |   recall@10 |   ndcg@10 |   recall@50 |   ndcg@50 |
|:---------------------------|-------:|:-------------------|--------:|------------:|----------:|------------:|----------:|
| clean_amazon_reduced_min10 |     42 | AsymLightGCN       |     410 |    0.062407 |  0.035040 |    0.151694 |  0.055458 |
| clean_amazon_reduced_min10 |     42 | LightGCN           |     410 |    0.012195 |  0.004790 |    0.026090 |  0.007861 |
| clean_amazon_reduced_min10 |     42 | MostPopIndependent |     410 |    0.036585 |  0.017192 |    0.092480 |  0.029472 |
| clean_amazon_reduced_min10 |     42 | NCL                |     410 |    0.014634 |  0.010494 |    0.026090 |  0.013161 |
| clean_amazon_reduced_min10 |     42 | Semantic-only      |     410 |    0.025203 |  0.011375 |    0.070463 |  0.022524 |
| clean_amazon_reduced_min10 |     42 | SemanticGatedBPR   |     410 |    0.036317 |  0.022981 |    0.120431 |  0.041443 |
| clean_amazon_reduced_min10 |     43 | AsymLightGCN       |     410 |    0.050961 |  0.023476 |    0.139597 |  0.043939 |
| clean_amazon_reduced_min10 |     43 | LightGCN           |     410 |    0.014634 |  0.009285 |    0.036179 |  0.014109 |
| clean_amazon_reduced_min10 |     43 | MostPopIndependent |     410 |    0.020935 |  0.012053 |    0.088618 |  0.026599 |
| clean_amazon_reduced_min10 |     43 | NCL                |     410 |    0.022764 |  0.013840 |    0.038618 |  0.017254 |
| clean_amazon_reduced_min10 |     43 | Semantic-only      |     410 |    0.014228 |  0.008288 |    0.081375 |  0.023646 |
| clean_amazon_reduced_min10 |     43 | SemanticGatedBPR   |     410 |    0.038278 |  0.021959 |    0.100898 |  0.035505 |
| clean_amazon_reduced_min10 |     44 | AsymLightGCN       |     410 |    0.038211 |  0.022648 |    0.134797 |  0.044930 |
| clean_amazon_reduced_min10 |     44 | LightGCN           |     410 |    0.004268 |  0.003321 |    0.021748 |  0.007129 |
| clean_amazon_reduced_min10 |     44 | MostPopIndependent |     410 |    0.020732 |  0.013652 |    0.060714 |  0.022402 |
| clean_amazon_reduced_min10 |     44 | NCL                |     410 |    0.009146 |  0.005229 |    0.034268 |  0.011160 |
| clean_amazon_reduced_min10 |     44 | Semantic-only      |     410 |    0.007666 |  0.004730 |    0.038153 |  0.011535 |
| clean_amazon_reduced_min10 |     44 | SemanticGatedBPR   |     410 |    0.046870 |  0.025621 |    0.082236 |  0.033615 |
| clean_amazon_reduced_min4  |     42 | AsymLightGCN       |    2157 |    0.044120 |  0.020506 |    0.131664 |  0.039813 |
| clean_amazon_reduced_min4  |     42 | LightGCN           |    2157 |    0.049065 |  0.028804 |    0.110841 |  0.042275 |
| clean_amazon_reduced_min4  |     42 | MostPopIndependent |    2157 |    0.032066 |  0.015264 |    0.102612 |  0.030862 |
| clean_amazon_reduced_min4  |     42 | NCL                |    2157 |    0.048911 |  0.029287 |    0.106066 |  0.041773 |
| clean_amazon_reduced_min4  |     42 | Semantic-only      |    2157 |    0.053701 |  0.037271 |    0.091474 |  0.045586 |
| clean_amazon_reduced_min4  |     42 | SemanticGatedBPR   |    2157 |    0.038339 |  0.017698 |    0.117616 |  0.034898 |
| clean_amazon_reduced_min4  |     43 | AsymLightGCN       |    2157 |    0.043556 |  0.021756 |    0.132907 |  0.041137 |
| clean_amazon_reduced_min4  |     43 | LightGCN           |    2157 |    0.047288 |  0.026800 |    0.106251 |  0.039879 |
| clean_amazon_reduced_min4  |     43 | MostPopIndependent |    2157 |    0.034346 |  0.016178 |    0.099574 |  0.030523 |
| clean_amazon_reduced_min4  |     43 | NCL                |    2157 |    0.048370 |  0.028444 |    0.101194 |  0.040030 |
| clean_amazon_reduced_min4  |     43 | Semantic-only      |    2157 |    0.056869 |  0.040675 |    0.094786 |  0.049015 |
| clean_amazon_reduced_min4  |     43 | SemanticGatedBPR   |    2157 |    0.031423 |  0.014719 |    0.105492 |  0.031289 |
| clean_amazon_reduced_min4  |     44 | AsymLightGCN       |    2157 |    0.040450 |  0.021409 |    0.135580 |  0.042027 |
| clean_amazon_reduced_min4  |     44 | LightGCN           |    2157 |    0.045511 |  0.027723 |    0.108214 |  0.041473 |
| clean_amazon_reduced_min4  |     44 | MostPopIndependent |    2157 |    0.032259 |  0.015587 |    0.096623 |  0.029673 |
| clean_amazon_reduced_min4  |     44 | NCL                |    2157 |    0.047365 |  0.028112 |    0.103964 |  0.040643 |
| clean_amazon_reduced_min4  |     44 | Semantic-only      |    2157 |    0.061582 |  0.041119 |    0.097921 |  0.048950 |
| clean_amazon_reduced_min4  |     44 | SemanticGatedBPR   |    2157 |    0.036277 |  0.018135 |    0.106668 |  0.033226 |
| clean_amazon_reduced_min6  |     42 | AsymLightGCN       |    1030 |    0.038867 |  0.018358 |    0.122391 |  0.036564 |
| clean_amazon_reduced_min6  |     42 | LightGCN           |    1030 |    0.023706 |  0.011791 |    0.068967 |  0.021389 |
| clean_amazon_reduced_min6  |     42 | MostPopIndependent |    1030 |    0.023463 |  0.010214 |    0.087760 |  0.023802 |
| clean_amazon_reduced_min6  |     42 | NCL                |    1030 |    0.021602 |  0.011661 |    0.076387 |  0.023527 |
| clean_amazon_reduced_min6  |     42 | Semantic-only      |    1030 |    0.027184 |  0.017401 |    0.070857 |  0.026807 |
| clean_amazon_reduced_min6  |     42 | SemanticGatedBPR   |    1030 |    0.030612 |  0.015970 |    0.086524 |  0.028377 |
| clean_amazon_reduced_min6  |     43 | AsymLightGCN       |    1030 |    0.046417 |  0.027376 |    0.124879 |  0.044971 |
| clean_amazon_reduced_min6  |     43 | LightGCN           |    1030 |    0.025081 |  0.012394 |    0.080559 |  0.024999 |
| clean_amazon_reduced_min6  |     43 | MostPopIndependent |    1030 |    0.035275 |  0.016408 |    0.096983 |  0.029691 |
| clean_amazon_reduced_min6  |     43 | NCL                |    1030 |    0.026052 |  0.013509 |    0.092938 |  0.028553 |
| clean_amazon_reduced_min6  |     43 | Semantic-only      |    1030 |    0.030259 |  0.020451 |    0.070469 |  0.029136 |
| clean_amazon_reduced_min6  |     43 | SemanticGatedBPR   |    1030 |    0.039644 |  0.021516 |    0.101088 |  0.034708 |
| clean_amazon_reduced_min6  |     44 | AsymLightGCN       |    1030 |    0.041939 |  0.021137 |    0.127699 |  0.039519 |
| clean_amazon_reduced_min6  |     44 | LightGCN           |    1030 |    0.026262 |  0.016038 |    0.084491 |  0.028423 |
| clean_amazon_reduced_min6  |     44 | MostPopIndependent |    1030 |    0.027023 |  0.013639 |    0.088592 |  0.026997 |
| clean_amazon_reduced_min6  |     44 | NCL                |    1030 |    0.027323 |  0.016955 |    0.083035 |  0.028768 |
| clean_amazon_reduced_min6  |     44 | Semantic-only      |    1030 |    0.040453 |  0.025694 |    0.083738 |  0.035081 |
| clean_amazon_reduced_min6  |     44 | SemanticGatedBPR   |    1030 |    0.033657 |  0.014485 |    0.099816 |  0.028827 |
| clean_amazon_reduced_min8  |     42 | AsymLightGCN       |     602 |    0.034474 |  0.017510 |    0.112160 |  0.034812 |
| clean_amazon_reduced_min8  |     42 | LightGCN           |     602 |    0.017027 |  0.008229 |    0.045266 |  0.014541 |
| clean_amazon_reduced_min8  |     42 | MostPopIndependent |     602 |    0.023533 |  0.012223 |    0.085875 |  0.026058 |
| clean_amazon_reduced_min8  |     42 | NCL                |     602 |    0.017857 |  0.010110 |    0.051545 |  0.017389 |
| clean_amazon_reduced_min8  |     42 | Semantic-only      |     602 |    0.037652 |  0.019776 |    0.069209 |  0.026761 |
| clean_amazon_reduced_min8  |     42 | SemanticGatedBPR   |     602 |    0.042902 |  0.020551 |    0.101970 |  0.033634 |
| clean_amazon_reduced_min8  |     43 | AsymLightGCN       |     602 |    0.056008 |  0.030707 |    0.148910 |  0.051801 |
| clean_amazon_reduced_min8  |     43 | LightGCN           |     602 |    0.013289 |  0.005794 |    0.035675 |  0.010712 |
| clean_amazon_reduced_min8  |     43 | MostPopIndependent |     602 |    0.032392 |  0.016329 |    0.090769 |  0.029344 |
| clean_amazon_reduced_min8  |     43 | NCL                |     602 |    0.024917 |  0.011618 |    0.051218 |  0.017689 |
| clean_amazon_reduced_min8  |     43 | Semantic-only      |     602 |    0.023810 |  0.012466 |    0.063206 |  0.021235 |
| clean_amazon_reduced_min8  |     43 | SemanticGatedBPR   |     602 |    0.039082 |  0.020448 |    0.116786 |  0.037879 |
| clean_amazon_reduced_min8  |     44 | AsymLightGCN       |     602 |    0.051218 |  0.029151 |    0.127497 |  0.046622 |
| clean_amazon_reduced_min8  |     44 | LightGCN           |     602 |    0.012625 |  0.007069 |    0.033112 |  0.011596 |
| clean_amazon_reduced_min8  |     44 | MostPopIndependent |     602 |    0.030454 |  0.015481 |    0.067691 |  0.024397 |
| clean_amazon_reduced_min8  |     44 | NCL                |     602 |    0.021678 |  0.010475 |    0.056091 |  0.017950 |
| clean_amazon_reduced_min8  |     44 | Semantic-only      |     602 |    0.039729 |  0.022629 |    0.081949 |  0.032366 |
| clean_amazon_reduced_min8  |     44 | SemanticGatedBPR   |     602 |    0.035815 |  0.017381 |    0.099796 |  0.031719 |

## Paired User-Level Tests

| Dataset                    | Comparison                        | Metric    |   Paired Seed-User Rows |   Users |   Mean Delta |   Median Delta |   Bootstrap 95% CI Low |   Bootstrap 95% CI High | Wilcoxon p   | Sign p   |   Positive Users |   Negative Users |   Zero Users |
|:---------------------------|:----------------------------------|:----------|------------------------:|--------:|-------------:|---------------:|-----------------------:|------------------------:|:-------------|:---------|-----------------:|-----------------:|-------------:|
| clean_amazon_reduced_min4  | AsymLightGCN - LightGCN           | recall@10 |                    6471 |    2157 |    -0.004579 |              0 |              -0.011096 |                0.002156 | 0.1908       | 0.7329   |              151 |              158 |         1848 |
| clean_amazon_reduced_min4  | AsymLightGCN - LightGCN           | ndcg@10   |                    6471 |    2157 |    -0.006552 |              0 |              -0.010839 |               -0.002214 | 0.0192       | 0.2871   |              166 |              187 |         1804 |
| clean_amazon_reduced_min4  | AsymLightGCN - LightGCN           | recall@50 |                    6471 |    2157 |     0.024949 |              0 |               0.015593 |                0.034334 | <0.0001      | <0.0001  |              369 |              233 |         1555 |
| clean_amazon_reduced_min4  | AsymLightGCN - LightGCN           | ndcg@50   |                    6471 |    2157 |    -0.000217 |              0 |              -0.004481 |                0.0039   | 0.1205       | 0.0152   |              427 |              358 |         1372 |
| clean_amazon_reduced_min4  | AsymLightGCN - NCL                | recall@10 |                    6471 |    2157 |    -0.005507 |              0 |              -0.012121 |                0.001275 | 0.1214       | 0.5720   |              151 |              162 |         1844 |
| clean_amazon_reduced_min4  | AsymLightGCN - NCL                | ndcg@10   |                    6471 |    2157 |    -0.007391 |              0 |              -0.011737 |               -0.003195 | 0.0086       | 0.2228   |              166 |              190 |         1801 |
| clean_amazon_reduced_min4  | AsymLightGCN - NCL                | recall@50 |                    6471 |    2157 |     0.029643 |              0 |               0.019865 |                0.039162 | <0.0001      | <0.0001  |              387 |              239 |         1531 |
| clean_amazon_reduced_min4  | AsymLightGCN - NCL                | ndcg@50   |                    6471 |    2157 |     0.000177 |              0 |              -0.004082 |                0.004466 | 0.0500       | 0.0069   |              435 |              358 |         1364 |
| clean_amazon_reduced_min4  | AsymLightGCN - SemanticGatedBPR   | recall@10 |                    6471 |    2157 |     0.007362 |              0 |               0.002787 |                0.011978 | 0.0017       | 0.0094   |              121 |               83 |         1953 |
| clean_amazon_reduced_min4  | AsymLightGCN - SemanticGatedBPR   | ndcg@10   |                    6471 |    2157 |     0.004373 |              0 |               0.00185  |                0.006968 | 0.0006       | 0.0010   |              173 |              116 |         1868 |
| clean_amazon_reduced_min4  | AsymLightGCN - SemanticGatedBPR   | recall@50 |                    6471 |    2157 |     0.023459 |              0 |               0.015834 |                0.03134  | <0.0001      | <0.0001  |              275 |              160 |         1722 |
| clean_amazon_reduced_min4  | AsymLightGCN - SemanticGatedBPR   | ndcg@50   |                    6471 |    2157 |     0.007855 |              0 |               0.00519  |                0.010584 | <0.0001      | <0.0001  |              442 |              295 |         1420 |
| clean_amazon_reduced_min4  | AsymLightGCN - Semantic-only      | recall@10 |                    6471 |    2157 |    -0.014676 |              0 |              -0.022905 |               -0.006168 | 0.0002       | 0.1081   |              182 |              215 |         1760 |
| clean_amazon_reduced_min4  | AsymLightGCN - Semantic-only      | ndcg@10   |                    6471 |    2157 |    -0.018465 |              0 |              -0.024502 |               -0.012703 | <0.0001      | 0.0291   |              190 |              236 |         1731 |
| clean_amazon_reduced_min4  | AsymLightGCN - Semantic-only      | recall@50 |                    6471 |    2157 |     0.038657 |              0 |               0.027393 |                0.050101 | <0.0001      | <0.0001  |              470 |              272 |         1415 |
| clean_amazon_reduced_min4  | AsymLightGCN - Semantic-only      | ndcg@50   |                    6471 |    2157 |    -0.006858 |              0 |              -0.01258  |               -0.001068 | 0.3685       | <0.0001  |              496 |              350 |         1311 |
| clean_amazon_reduced_min4  | AsymLightGCN - MostPopIndependent | recall@10 |                    6471 |    2157 |     0.009818 |              0 |               0.004651 |                0.015047 | 0.0004       | 0.0011   |              133 |               84 |         1940 |
| clean_amazon_reduced_min4  | AsymLightGCN - MostPopIndependent | ndcg@10   |                    6471 |    2157 |     0.005547 |              0 |               0.003175 |                0.008011 | <0.0001      | 0.0037   |              168 |              118 |         1871 |
| clean_amazon_reduced_min4  | AsymLightGCN - MostPopIndependent | recall@50 |                    6471 |    2157 |     0.033781 |              0 |               0.025554 |                0.041981 | <0.0001      | <0.0001  |              310 |              148 |         1699 |
| clean_amazon_reduced_min4  | AsymLightGCN - MostPopIndependent | ndcg@50   |                    6471 |    2157 |     0.01064  |              0 |               0.007915 |                0.013399 | <0.0001      | <0.0001  |              446 |              273 |         1438 |
| clean_amazon_reduced_min6  | AsymLightGCN - LightGCN           | recall@10 |                    3090 |    1030 |     0.017391 |              0 |               0.009143 |                0.025856 | <0.0001      | <0.0001  |               86 |               38 |          906 |
| clean_amazon_reduced_min6  | AsymLightGCN - LightGCN           | ndcg@10   |                    3090 |    1030 |     0.008883 |              0 |               0.003962 |                0.01369  | <0.0001      | <0.0001  |              101 |               51 |          878 |
| clean_amazon_reduced_min6  | AsymLightGCN - LightGCN           | recall@50 |                    3090 |    1030 |     0.046984 |              0 |               0.032592 |                0.060815 | <0.0001      | <0.0001  |              219 |              107 |          704 |
| clean_amazon_reduced_min6  | AsymLightGCN - LightGCN           | ndcg@50   |                    3090 |    1030 |     0.015415 |              0 |               0.010247 |                0.020885 | <0.0001      | <0.0001  |              255 |              141 |          634 |
| clean_amazon_reduced_min6  | AsymLightGCN - NCL                | recall@10 |                    3090 |    1030 |     0.017415 |              0 |               0.009081 |                0.02574  | 0.0001       | <0.0001  |               88 |               38 |          904 |
| clean_amazon_reduced_min6  | AsymLightGCN - NCL                | ndcg@10   |                    3090 |    1030 |     0.008249 |              0 |               0.003423 |                0.013239 | 0.0006       | 0.0006   |               98 |               55 |          877 |
| clean_amazon_reduced_min6  | AsymLightGCN - NCL                | recall@50 |                    3090 |    1030 |     0.04087  |              0 |               0.026964 |                0.055441 | <0.0001      | <0.0001  |              209 |              110 |          711 |
| clean_amazon_reduced_min6  | AsymLightGCN - NCL                | ndcg@50   |                    3090 |    1030 |     0.013402 |              0 |               0.008061 |                0.018845 | <0.0001      | <0.0001  |              250 |              146 |          634 |
| clean_amazon_reduced_min6  | AsymLightGCN - SemanticGatedBPR   | recall@10 |                    3090 |    1030 |     0.00777  |              0 |               0.001035 |                0.014518 | 0.0388       | 0.0134   |               60 |               35 |          935 |
| clean_amazon_reduced_min6  | AsymLightGCN - SemanticGatedBPR   | ndcg@10   |                    3090 |    1030 |     0.004967 |              0 |               0.001288 |                0.008813 | 0.0305       | 0.0036   |               91 |               55 |          884 |
| clean_amazon_reduced_min6  | AsymLightGCN - SemanticGatedBPR   | recall@50 |                    3090 |    1030 |     0.029181 |              0 |               0.017518 |                0.041371 | <0.0001      | <0.0001  |              173 |               86 |          771 |
| clean_amazon_reduced_min6  | AsymLightGCN - SemanticGatedBPR   | ndcg@50   |                    3090 |    1030 |     0.009714 |              0 |               0.005695 |                0.013963 | <0.0001      | <0.0001  |              236 |              143 |          651 |
| clean_amazon_reduced_min6  | AsymLightGCN - Semantic-only      | recall@10 |                    3090 |    1030 |     0.009775 |              0 |              -0.000133 |                0.019677 | 0.0563       | 0.0026   |              100 |               61 |          869 |
| clean_amazon_reduced_min6  | AsymLightGCN - Semantic-only      | ndcg@10   |                    3090 |    1030 |     0.001108 |              0 |              -0.00573  |                0.007685 | 0.2446       | 0.0042   |              108 |               69 |          853 |
| clean_amazon_reduced_min6  | AsymLightGCN - Semantic-only      | recall@50 |                    3090 |    1030 |     0.049969 |              0 |               0.035391 |                0.064312 | <0.0001      | <0.0001  |              244 |              102 |          684 |
| clean_amazon_reduced_min6  | AsymLightGCN - Semantic-only      | ndcg@50   |                    3090 |    1030 |     0.01001  |              0 |               0.003325 |                0.016466 | <0.0001      | <0.0001  |              274 |              135 |          621 |
| clean_amazon_reduced_min6  | AsymLightGCN - MostPopIndependent | recall@10 |                    3090 |    1030 |     0.013821 |              0 |               0.006163 |                0.021338 | 0.0006       | 0.0002   |               67 |               30 |          933 |
| clean_amazon_reduced_min6  | AsymLightGCN - MostPopIndependent | ndcg@10   |                    3090 |    1030 |     0.00887  |              0 |               0.005051 |                0.013113 | 0.0001       | 0.0001   |               87 |               43 |          900 |
| clean_amazon_reduced_min6  | AsymLightGCN - MostPopIndependent | recall@50 |                    3090 |    1030 |     0.033878 |              0 |               0.02015  |                0.047927 | <0.0001      | <0.0001  |              191 |              106 |          733 |
| clean_amazon_reduced_min6  | AsymLightGCN - MostPopIndependent | ndcg@50   |                    3090 |    1030 |     0.013521 |              0 |               0.008712 |                0.018537 | <0.0001      | <0.0001  |              240 |              151 |          639 |
| clean_amazon_reduced_min8  | AsymLightGCN - LightGCN           | recall@10 |                    1806 |     602 |     0.03292  |              0 |               0.021592 |                0.045022 | <0.0001      | <0.0001  |               74 |               19 |          509 |
| clean_amazon_reduced_min8  | AsymLightGCN - LightGCN           | ndcg@10   |                    1806 |     602 |     0.018758 |              0 |               0.012361 |                0.025823 | <0.0001      | <0.0001  |               77 |               19 |          506 |
| clean_amazon_reduced_min8  | AsymLightGCN - LightGCN           | recall@50 |                    1806 |     602 |     0.091505 |              0 |               0.074085 |                0.1096   | <0.0001      | <0.0001  |              172 |               35 |          395 |
| clean_amazon_reduced_min8  | AsymLightGCN - LightGCN           | ndcg@50   |                    1806 |     602 |     0.032129 |              0 |               0.02488  |                0.039553 | <0.0001      | <0.0001  |              185 |               43 |          374 |
| clean_amazon_reduced_min8  | AsymLightGCN - NCL                | recall@10 |                    1806 |     602 |     0.025749 |              0 |               0.01384  |                0.037563 | <0.0001      | <0.0001  |               70 |               27 |          505 |
| clean_amazon_reduced_min8  | AsymLightGCN - NCL                | ndcg@10   |                    1806 |     602 |     0.015055 |              0 |               0.008584 |                0.021968 | <0.0001      | <0.0001  |               73 |               30 |          499 |
| clean_amazon_reduced_min8  | AsymLightGCN - NCL                | recall@50 |                    1806 |     602 |     0.076571 |              0 |               0.058853 |                0.094706 | <0.0001      | <0.0001  |              155 |               43 |          404 |
| clean_amazon_reduced_min8  | AsymLightGCN - NCL                | ndcg@50   |                    1806 |     602 |     0.026736 |              0 |               0.019622 |                0.034407 | <0.0001      | <0.0001  |              172 |               60 |          370 |
| clean_amazon_reduced_min8  | AsymLightGCN - SemanticGatedBPR   | recall@10 |                    1806 |     602 |     0.007967 |              0 |              -0.002708 |                0.018731 | 0.1915       | 1.0000   |               46 |               45 |          511 |
| clean_amazon_reduced_min8  | AsymLightGCN - SemanticGatedBPR   | ndcg@10   |                    1806 |     602 |     0.006329 |              0 |               0.000285 |                0.012388 | 0.0791       | 0.8522   |               59 |               56 |          487 |
| clean_amazon_reduced_min8  | AsymLightGCN - SemanticGatedBPR   | recall@50 |                    1806 |     602 |     0.023338 |              0 |               0.008729 |                0.037838 | 0.0019       | 0.0216   |               95 |               65 |          442 |
| clean_amazon_reduced_min8  | AsymLightGCN - SemanticGatedBPR   | ndcg@50   |                    1806 |     602 |     0.010001 |              0 |               0.003961 |                0.016502 | 0.0051       | 0.1250   |              135 |              110 |          357 |
| clean_amazon_reduced_min8  | AsymLightGCN - Semantic-only      | recall@10 |                    1806 |     602 |     0.013503 |              0 |              -6.3e-05  |                0.02663  | 0.0401       | 0.0046   |               72 |               41 |          489 |
| clean_amazon_reduced_min8  | AsymLightGCN - Semantic-only      | ndcg@10   |                    1806 |     602 |     0.007499 |              0 |              -0.000171 |                0.015334 | 0.0222       | 0.0012   |               77 |               41 |          484 |
| clean_amazon_reduced_min8  | AsymLightGCN - Semantic-only      | recall@50 |                    1806 |     602 |     0.058068 |              0 |               0.039432 |                0.07743  | <0.0001      | <0.0001  |              156 |               65 |          381 |
| clean_amazon_reduced_min8  | AsymLightGCN - Semantic-only      | ndcg@50   |                    1806 |     602 |     0.017625 |              0 |               0.009253 |                0.025636 | <0.0001      | <0.0001  |              174 |               81 |          347 |
| clean_amazon_reduced_min8  | AsymLightGCN - MostPopIndependent | recall@10 |                    1806 |     602 |     0.01844  |              0 |               0.007135 |                0.030396 | 0.0021       | 0.0009   |               57 |               26 |          519 |
| clean_amazon_reduced_min8  | AsymLightGCN - MostPopIndependent | ndcg@10   |                    1806 |     602 |     0.011111 |              0 |               0.004373 |                0.018094 | 0.0018       | 0.0051   |               65 |               36 |          501 |
| clean_amazon_reduced_min8  | AsymLightGCN - MostPopIndependent | recall@50 |                    1806 |     602 |     0.048077 |              0 |               0.03105  |                0.064964 | <0.0001      | <0.0001  |              117 |               50 |          435 |
| clean_amazon_reduced_min8  | AsymLightGCN - MostPopIndependent | ndcg@50   |                    1806 |     602 |     0.017812 |              0 |               0.010668 |                0.025567 | <0.0001      | 0.0029   |              138 |               92 |          372 |
| clean_amazon_reduced_min10 | AsymLightGCN - LightGCN           | recall@10 |                    1230 |     410 |     0.04016  |              0 |               0.027121 |                0.053841 | <0.0001      | <0.0001  |               59 |                8 |          343 |
| clean_amazon_reduced_min10 | AsymLightGCN - LightGCN           | ndcg@10   |                    1230 |     410 |     0.021256 |              0 |               0.014073 |                0.029015 | <0.0001      | <0.0001  |               60 |               11 |          339 |
| clean_amazon_reduced_min10 | AsymLightGCN - LightGCN           | recall@50 |                    1230 |     410 |     0.114024 |              0 |               0.091863 |                0.137877 | <0.0001      | <0.0001  |              132 |               21 |          257 |
| clean_amazon_reduced_min10 | AsymLightGCN - LightGCN           | ndcg@50   |                    1230 |     410 |     0.038409 |              0 |               0.029561 |                0.047879 | <0.0001      | <0.0001  |              137 |               27 |          246 |
| clean_amazon_reduced_min10 | AsymLightGCN - NCL                | recall@10 |                    1230 |     410 |     0.035011 |              0 |               0.021447 |                0.048949 | <0.0001      | <0.0001  |               58 |               14 |          338 |
| clean_amazon_reduced_min10 | AsymLightGCN - NCL                | ndcg@10   |                    1230 |     410 |     0.0172   |              0 |               0.009057 |                0.025546 | <0.0001      | <0.0001  |               59 |               17 |          334 |
| clean_amazon_reduced_min10 | AsymLightGCN - NCL                | recall@50 |                    1230 |     410 |     0.109037 |              0 |               0.087764 |                0.131997 | <0.0001      | <0.0001  |              127 |               18 |          265 |
| clean_amazon_reduced_min10 | AsymLightGCN - NCL                | ndcg@50   |                    1230 |     410 |     0.034251 |              0 |               0.024615 |                0.043805 | <0.0001      | <0.0001  |              132 |               28 |          250 |
| clean_amazon_reduced_min10 | AsymLightGCN - SemanticGatedBPR   | recall@10 |                    1230 |     410 |     0.010038 |              0 |              -0.003807 |                0.023369 | 0.1463       | 0.3203   |               46 |               36 |          328 |
| clean_amazon_reduced_min10 | AsymLightGCN - SemanticGatedBPR   | ndcg@10   |                    1230 |     410 |     0.003534 |              0 |              -0.004668 |                0.011321 | 0.4383       | 0.4657   |               50 |               42 |          318 |
| clean_amazon_reduced_min10 | AsymLightGCN - SemanticGatedBPR   | recall@50 |                    1230 |     410 |     0.040841 |              0 |               0.02122  |                0.060675 | <0.0001      | 0.0014   |               87 |               49 |          274 |
| clean_amazon_reduced_min10 | AsymLightGCN - SemanticGatedBPR   | ndcg@50   |                    1230 |     410 |     0.011254 |              0 |               0.00333  |                0.019378 | 0.0058       | 0.0097   |              110 |               74 |          226 |
| clean_amazon_reduced_min10 | AsymLightGCN - Semantic-only      | recall@10 |                    1230 |     410 |     0.034828 |              0 |               0.021137 |                0.049364 | <0.0001      | <0.0001  |               60 |               17 |          333 |
| clean_amazon_reduced_min10 | AsymLightGCN - Semantic-only      | ndcg@10   |                    1230 |     410 |     0.018924 |              0 |               0.011624 |                0.02655  | <0.0001      | <0.0001  |               64 |               16 |          330 |
| clean_amazon_reduced_min10 | AsymLightGCN - Semantic-only      | recall@50 |                    1230 |     410 |     0.078699 |              0 |               0.055718 |                0.102628 | <0.0001      | <0.0001  |              119 |               43 |          248 |
| clean_amazon_reduced_min10 | AsymLightGCN - Semantic-only      | ndcg@50   |                    1230 |     410 |     0.028874 |              0 |               0.020871 |                0.037416 | <0.0001      | <0.0001  |              132 |               50 |          228 |
| clean_amazon_reduced_min10 | AsymLightGCN - MostPopIndependent | recall@10 |                    1230 |     410 |     0.024442 |              0 |               0.010414 |                0.038987 | 0.0010       | 0.0003   |               51 |               20 |          339 |
| clean_amazon_reduced_min10 | AsymLightGCN - MostPopIndependent | ndcg@10   |                    1230 |     410 |     0.012756 |              0 |               0.004584 |                0.021257 | 0.0017       | 0.0011   |               55 |               25 |          330 |
| clean_amazon_reduced_min10 | AsymLightGCN - MostPopIndependent | recall@50 |                    1230 |     410 |     0.061425 |              0 |               0.039484 |                0.084391 | <0.0001      | <0.0001  |               95 |               38 |          277 |
| clean_amazon_reduced_min10 | AsymLightGCN - MostPopIndependent | ndcg@50   |                    1230 |     410 |     0.021951 |              0 |               0.01328  |                0.031115 | <0.0001      | <0.0001  |              115 |               61 |          234 |

## Checkpoints

### amazon_reduced_min4

- seed 42: AsymLightGCN=LightGCN-May-14-2026_09-57-22.pth; LightGCN=LightGCN-May-14-2026_09-57-07.pth; NCL=NCL-May-14-2026_09-57-40.pth; SemanticGatedBPR=BPR-May-14-2026_09-53-56.pth
- seed 43: AsymLightGCN=LightGCN-May-14-2026_09-59-21.pth; LightGCN=LightGCN-May-14-2026_09-59-06.pth; NCL=NCL-May-14-2026_09-59-39.pth; SemanticGatedBPR=BPR-May-14-2026_09-54-59.pth
- seed 44: AsymLightGCN=LightGCN-May-14-2026_10-01-20.pth; LightGCN=LightGCN-May-14-2026_10-01-05.pth; NCL=NCL-May-14-2026_10-01-38.pth; SemanticGatedBPR=BPR-May-14-2026_09-56-04.pth
### amazon_reduced_min6

- seed 42: AsymLightGCN=LightGCN-May-14-2026_09-58-08.pth; LightGCN=LightGCN-May-14-2026_09-58-00.pth; NCL=NCL-May-14-2026_09-58-17.pth; SemanticGatedBPR=BPR-May-14-2026_09-54-20.pth
- seed 43: AsymLightGCN=LightGCN-May-14-2026_10-00-07.pth; LightGCN=LightGCN-May-14-2026_09-59-59.pth; NCL=NCL-May-14-2026_10-00-17.pth; SemanticGatedBPR=BPR-May-14-2026_09-55-21.pth
- seed 44: AsymLightGCN=LightGCN-May-14-2026_10-02-06.pth; LightGCN=LightGCN-May-14-2026_10-01-58.pth; NCL=NCL-May-14-2026_10-02-15.pth; SemanticGatedBPR=BPR-May-14-2026_09-56-30.pth
### amazon_reduced_min8

- seed 42: AsymLightGCN=LightGCN-May-14-2026_09-58-34.pth; LightGCN=LightGCN-May-14-2026_09-58-29.pth; NCL=NCL-May-14-2026_09-58-40.pth; SemanticGatedBPR=BPR-May-14-2026_09-54-33.pth
- seed 43: AsymLightGCN=LightGCN-May-14-2026_10-00-33.pth; LightGCN=LightGCN-May-14-2026_10-00-29.pth; NCL=NCL-May-14-2026_10-00-39.pth; SemanticGatedBPR=BPR-May-14-2026_09-55-35.pth
- seed 44: AsymLightGCN=LightGCN-May-14-2026_10-02-33.pth; LightGCN=LightGCN-May-14-2026_10-02-28.pth; NCL=NCL-May-14-2026_10-02-38.pth; SemanticGatedBPR=BPR-May-14-2026_09-56-43.pth
### amazon_reduced_min10

- seed 42: AsymLightGCN=LightGCN-May-14-2026_09-58-52.pth; LightGCN=LightGCN-May-14-2026_09-58-48.pth; NCL=NCL-May-14-2026_09-58-56.pth; SemanticGatedBPR=BPR-May-14-2026_09-54-40.pth
- seed 43: AsymLightGCN=LightGCN-May-14-2026_10-00-52.pth; LightGCN=LightGCN-May-14-2026_10-00-48.pth; NCL=NCL-May-14-2026_10-00-56.pth; SemanticGatedBPR=BPR-May-14-2026_09-55-44.pth
- seed 44: AsymLightGCN=LightGCN-May-14-2026_10-02-51.pth; LightGCN=LightGCN-May-14-2026_10-02-47.pth; NCL=NCL-May-14-2026_10-02-55.pth; SemanticGatedBPR=BPR-May-14-2026_09-56-51.pth
