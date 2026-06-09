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
| BAAI/bge-small-en-v1.5 | 0.9770 | 0.9904 | +0.0134 | 0.9929 |
| intfloat/e5-small-v2 | 0.9846 | 0.9916 | +0.0070 | 0.9939 |
| sentence-transformers/all-MiniLM-L12-v2 | 0.9741 | 0.9938 | +0.0196 | 0.9946 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.9886 | 0.9947 | +0.0061 | 0.9960 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 0.9940 | 0.9985 | +0.0045 | 0.9979 |
| sentence-transformers/paraphrase-albert-small-v2 | 0.8857 | 0.9698 | +0.0840 | 0.9869 |

## Strategy Summary

| strategy | mean | min | max |
| --- | --- | --- | --- |
| farthest_row_zscore | 0.9674 | 0.8857 | 0.9940 |
| learned_top256 | 0.9898 | 0.9698 | 0.9985 |
| learned_weighted_bank | 0.9937 | 0.9869 | 0.9979 |
| random_row_zscore | 0.9559 | 0.8574 | 0.9889 |

## Relation-Only Top-10 Check

Pool 10 is the closest proxy here for relation-only top-10, because the relative signature retrieves exactly 10 candidates while raw embedding top-10 is treated as ground truth. Learned anchors can improve this case without making it a complete replacement for raw reranking.

| strategy | mean pool-10 recall | min | max |
| --- | --- | --- | --- |
| farthest_row_zscore | 0.5090 | 0.3844 | 0.5740 |
| learned_top256 | 0.5485 | 0.4547 | 0.6005 |
| learned_weighted_bank | 0.5414 | 0.4231 | 0.5901 |
| random_row_zscore | 0.4856 | 0.3545 | 0.5507 |

Mean recall by pool:

| strategy | pool 25 | pool 50 | pool 100 | pool 250 |
| --- | --- | --- | --- | --- |
| farthest_row_zscore | 0.7141 | 0.8292 | 0.9088 | 0.9674 |
| learned_top256 | 0.7731 | 0.8880 | 0.9535 | 0.9898 |
| learned_weighted_bank | 0.7690 | 0.8901 | 0.9589 | 0.9937 |
| random_row_zscore | 0.6838 | 0.8010 | 0.8858 | 0.9559 |

## Finding

`learned_top256` should be treated as a direct routing upgrade if it beats `farthest_row_zscore` on held-out queries. If the smallest-pool recall stays well below the larger-pool recall, relation-only top-k is still not strong enough to eliminate raw reranking.

## Artifacts

- CSV: `experiments/learned_anchor_selection_seed29/learned_anchor_selection.csv`
- Plot: `experiments/learned_anchor_selection_seed29/learned_anchor_selection_pool250.png`
