# Bilinear Relation Metric

## Scope

- Embedding file: `experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy`
- Documents: 10000
- Train docs: first 8000
- Eval queries: 1000 sampled from held-out later docs
- Anchors: 256 farthest base-document anchors
- Projection rank: 256
- Training steps: 1000

## Method

Fixed relation signatures are computed once. The baseline ranks by cosine in signature space. `contrastive_bilinear` trains a projection `project(sig) = normalize(sig @ L)` with raw-neighbor positives and hard relation negatives. `ridge_bilinear` distills raw geometry by solving a ridge regression from relation signatures to raw embeddings, then ranks by cosine in the projected relation-only space. Both learned methods are bilinear at scoring time: `score(q,d) = sig(q)^T W sig(d)`. Evaluation is relation-only: no raw embedding rerank is used.

## Recall by Candidate Count

| method | pool 10 | pool 25 | pool 50 | pool 100 | pool 250 |
| --- | --- | --- | --- | --- | --- |
| cosine_relation | 0.6236 | 0.8176 | 0.9033 | 0.9534 | 0.9841 |
| contrastive_bilinear | 0.5931 | 0.7826 | 0.8683 | 0.9215 | 0.9673 |
| ridge_bilinear | 0.8314 | 0.9846 | 0.9975 | 0.9995 | 1.0000 |

## Delta vs Cosine

| pool | cosine | bilinear | delta | bilinear all top-k |
| --- | --- | --- | --- | --- |
| 10 | 0.6236 | 0.8314 | +0.2078 | 0.1320 |
| 25 | 0.8176 | 0.9846 | +0.1670 | 0.8780 |
| 50 | 0.9033 | 0.9975 | +0.0942 | 0.9780 |
| 100 | 0.9534 | 0.9995 | +0.0461 | 0.9950 |
| 250 | 0.9841 | 1.0000 | +0.0159 | 1.0000 |

## Training

- Initial loss: 2.9504
- Final loss: 1.3520

## Finding

This directly tests whether correlations between anchor relations recover final relation-only top-k. A positive delta at pool 10 means the bilinear metric improves final top-k ordering, while high recall only at larger pools means it remains a routing improvement.

## Artifacts

- CSV: `experiments/bilinear_relation_metric/bilinear_relation_metric.csv`
- Contrastive projection: `experiments/bilinear_relation_metric/contrastive_bilinear_projection.npy`
- Ridge projection: `experiments/bilinear_relation_metric/ridge_bilinear_projection.npy`
- Plot: `experiments/bilinear_relation_metric/bilinear_relation_pool10.png`
