# Learned Anchor Selection

## Scope

- Models evaluated: 6
- Documents per model: 3000
- Base/old docs used for anchor candidates: 2000
- Held-out new-doc queries: 800
- Candidate anchor bank: 512
- Final selected anchors: 256
- Training steps per model: 250

## Method

This tests whether anchor selection can be trained as a separate lightweight model. For each embedding model, the script builds a larger farthest-point candidate anchor bank from the first/base documents, computes relative signatures to that bank, then trains one positive weight per candidate anchor with a contrastive hard-negative loss using only base-document positives and negatives. After training, it keeps the highest-weighted 256 anchors and evaluates on held-out queries from later documents against the full corpus.

Strategies:

- `random_row_zscore`: random 256 base-document anchors.
- `farthest_row_zscore`: 256 farthest-point base-document anchors.
- `learned_top256`: top 256 anchors selected by the trained anchor weights.
- `learned_weighted_bank`: all candidate anchors with learned weights, included as an upper-bound diagnostic rather than a fair storage match.

## Held-Out Results at Pool 250

| model | farthest | learned top256 | delta | weighted bank |
| --- | --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 | 0.9747 | 0.9915 | +0.0168 | 0.9930 |
| intfloat/e5-small-v2 | 0.9840 | 0.9890 | +0.0050 | 0.9916 |
| sentence-transformers/all-MiniLM-L12-v2 | 0.9776 | 0.9950 | +0.0174 | 0.9959 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.9894 | 0.9965 | +0.0071 | 0.9966 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 0.9930 | 0.9974 | +0.0044 | 0.9966 |
| sentence-transformers/paraphrase-albert-small-v2 | 0.8939 | 0.9726 | +0.0787 | 0.9845 |

## Strategy Summary

| strategy | mean | min | max |
| --- | --- | --- | --- |
| farthest_row_zscore | 0.9688 | 0.8939 | 0.9930 |
| learned_top256 | 0.9903 | 0.9726 | 0.9974 |
| learned_weighted_bank | 0.9930 | 0.9845 | 0.9966 |
| random_row_zscore | 0.9548 | 0.8624 | 0.9826 |

## Relation-Only Top-10 Check

Pool 10 is the closest proxy here for relation-only top-10, because the relative signature retrieves exactly 10 candidates while raw embedding top-10 is treated as ground truth. Learned anchors can improve this case without making it a complete replacement for raw reranking.

| strategy | mean pool-10 recall | min | max |
| --- | --- | --- | --- |
| farthest_row_zscore | 0.5106 | 0.3868 | 0.5761 |
| learned_top256 | 0.5505 | 0.4579 | 0.6012 |
| learned_weighted_bank | 0.5367 | 0.4096 | 0.5931 |
| random_row_zscore | 0.4789 | 0.3471 | 0.5360 |

Mean recall by pool:

| strategy | pool 25 | pool 50 | pool 100 | pool 250 |
| --- | --- | --- | --- | --- |
| farthest_row_zscore | 0.7161 | 0.8329 | 0.9097 | 0.9688 |
| learned_top256 | 0.7750 | 0.8898 | 0.9557 | 0.9903 |
| learned_weighted_bank | 0.7662 | 0.8870 | 0.9565 | 0.9930 |
| random_row_zscore | 0.6771 | 0.7950 | 0.8827 | 0.9548 |

## Finding

`learned_top256` should be treated as a direct routing upgrade if it beats `farthest_row_zscore` on held-out queries. If the smallest-pool recall stays well below the larger-pool recall, relation-only top-k is still not strong enough to eliminate raw reranking.

## Artifacts

- CSV: `experiments/learned_anchor_selection_seed17/learned_anchor_selection.csv`
- Plot: `experiments/learned_anchor_selection_seed17/learned_anchor_selection_pool250.png`
