# Ridge Bilinear Sweep

## Scope

- Embedding file: `experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy`
- Documents: 10000
- Train docs: 8000
- Eval queries: 1000
- Time budget minutes: 115.0

## Best Pool-10 Configs

| method | anchors | ridge | transform | target | recall | all top-k |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_bilinear | 1024 | 0.03 | row_zscore | raw | 0.9276 | 0.4350 |
| ridge_bilinear | 768 | 0.03 | row_zscore | raw | 0.9245 | 0.4200 |
| ridge_bilinear | 1024 | 0.01 | row_zscore | raw | 0.9239 | 0.4110 |
| ridge_bilinear | 1024 | 0.001 | raw_l2 | raw | 0.9229 | 0.3970 |
| ridge_bilinear | 1024 | 0.003 | row_zscore | raw | 0.9228 | 0.4020 |
| ridge_bilinear | 1024 | 0.1 | row_zscore | raw | 0.9224 | 0.4010 |
| ridge_bilinear | 768 | 0.01 | row_zscore | raw | 0.9221 | 0.4120 |
| ridge_bilinear | 768 | 0.001 | raw_l2 | raw | 0.9219 | 0.3960 |
| ridge_bilinear | 1024 | 0.001 | row_zscore | raw | 0.9218 | 0.3950 |
| ridge_bilinear | 768 | 0.1 | row_zscore | raw | 0.9208 | 0.4030 |

## Best Pool-25 Configs

| method | anchors | ridge | transform | target | recall | all top-k |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_bilinear | 1024 | 0.01 | row_zscore | raw | 0.9996 | 0.9960 |
| ridge_bilinear | 512 | 0.003 | row_zscore | raw | 0.9995 | 0.9950 |
| ridge_bilinear | 512 | 0.01 | row_zscore | raw | 0.9995 | 0.9950 |
| ridge_bilinear | 512 | 0.03 | row_zscore | raw | 0.9995 | 0.9950 |
| ridge_bilinear | 512 | 0.1 | row_zscore | raw | 0.9995 | 0.9950 |
| ridge_bilinear | 768 | 0.1 | row_zscore | raw | 0.9995 | 0.9950 |
| ridge_bilinear | 1024 | 0.003 | row_zscore | raw | 0.9995 | 0.9950 |
| ridge_bilinear | 1024 | 0.1 | row_zscore | raw | 0.9995 | 0.9950 |
| ridge_bilinear | 512 | 0.001 | row_zscore | raw | 0.9994 | 0.9940 |
| ridge_bilinear | 768 | 0.003 | row_zscore | raw | 0.9994 | 0.9940 |

## Finding

This sweep is designed to find whether relation-only top-k improves from anchor count, signature normalization, and ridge regularization. The most important number is pool-10 recall because that is the closest test of final top-k without raw reranking.

## Artifacts

- CSV: `experiments/ridge_bilinear_sweep/ridge_bilinear_sweep.csv`
- Plot: `experiments/ridge_bilinear_sweep/ridge_bilinear_sweep_pool10.png`
