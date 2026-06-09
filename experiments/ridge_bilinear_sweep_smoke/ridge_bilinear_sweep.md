# Ridge Bilinear Sweep

## Scope

- Embedding file: `experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy`
- Documents: 1500
- Train docs: 1000
- Eval queries: 100
- Time budget minutes: 5.0

## Best Pool-10 Configs

| method | anchors | ridge | transform | target | recall | all top-k |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_bilinear | 192 | 0.1 | row_zscore | raw | 0.7820 | 0.0600 |
| ridge_bilinear | 128 | 0.1 | row_zscore | raw | 0.7340 | 0.0300 |
| ridge_bilinear | 192 | 1.0 | row_zscore | raw | 0.6680 | 0.0500 |
| ridge_bilinear | 128 | 1.0 | row_zscore | raw | 0.6290 | 0.0300 |
| ridge_bilinear | 192 | 0.1 | row_col_zscore | raw | 0.5930 | 0.0200 |
| ridge_bilinear | 128 | 0.1 | row_col_zscore | raw | 0.5600 | 0.0200 |
| cosine_relation | 192 | 0.0 | row_zscore | none | 0.5550 | 0.0200 |
| cosine_relation | 192 | 0.0 | row_zscore | none | 0.5550 | 0.0200 |
| ridge_bilinear | 192 | 1.0 | row_col_zscore | raw | 0.5470 | 0.0200 |
| cosine_relation | 128 | 0.0 | row_zscore | none | 0.5210 | 0.0100 |

## Best Pool-25 Configs

| method | anchors | ridge | transform | target | recall | all top-k |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_bilinear | 192 | 0.1 | row_zscore | raw | 0.9710 | 0.7500 |
| ridge_bilinear | 128 | 0.1 | row_zscore | raw | 0.9400 | 0.5800 |
| ridge_bilinear | 192 | 1.0 | row_zscore | raw | 0.8960 | 0.4000 |
| ridge_bilinear | 128 | 1.0 | row_zscore | raw | 0.8510 | 0.2900 |
| ridge_bilinear | 192 | 0.1 | row_col_zscore | raw | 0.8060 | 0.2200 |
| ridge_bilinear | 128 | 0.1 | row_col_zscore | raw | 0.7890 | 0.2800 |
| cosine_relation | 192 | 0.0 | row_zscore | none | 0.7640 | 0.1600 |
| cosine_relation | 192 | 0.0 | row_zscore | none | 0.7640 | 0.1600 |
| ridge_bilinear | 192 | 1.0 | row_col_zscore | raw | 0.7510 | 0.2100 |
| ridge_bilinear | 128 | 1.0 | row_col_zscore | raw | 0.7340 | 0.1500 |

## Finding

This sweep is designed to find whether relation-only top-k improves from anchor count, signature normalization, and ridge regularization. The most important number is pool-10 recall because that is the closest test of final top-k without raw reranking.

## Artifacts

- CSV: `experiments/ridge_bilinear_sweep_smoke/ridge_bilinear_sweep.csv`
- Plot: `experiments/ridge_bilinear_sweep_smoke/ridge_bilinear_sweep_pool10.png`
