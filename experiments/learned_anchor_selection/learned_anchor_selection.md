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
| BAAI/bge-small-en-v1.5 | 0.9714 | 0.9891 | +0.0178 | 0.9915 |
| intfloat/e5-small-v2 | 0.9812 | 0.9926 | +0.0114 | 0.9920 |
| sentence-transformers/all-MiniLM-L12-v2 | 0.9725 | 0.9945 | +0.0220 | 0.9943 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.9878 | 0.9959 | +0.0081 | 0.9954 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 0.9924 | 0.9991 | +0.0067 | 0.9980 |
| sentence-transformers/paraphrase-albert-small-v2 | 0.8903 | 0.9699 | +0.0796 | 0.9906 |

## Strategy Summary

| strategy | mean | min | max |
| --- | --- | --- | --- |
| farthest_row_zscore | 0.9659 | 0.8903 | 0.9924 |
| learned_top256 | 0.9902 | 0.9699 | 0.9991 |
| learned_weighted_bank | 0.9936 | 0.9906 | 0.9980 |
| random_row_zscore | 0.9517 | 0.8588 | 0.9846 |

## Relation-Only Top-10 Check

Pool 10 is the closest proxy here for relation-only top-10, because the relative signature retrieves exactly 10 candidates while raw embedding top-10 is treated as ground truth. Learned anchors can improve this case without making it a complete replacement for raw reranking.

| strategy | mean pool-10 recall | min | max |
| --- | --- | --- | --- |
| farthest_row_zscore | 0.5080 | 0.3849 | 0.5682 |
| learned_top256 | 0.5511 | 0.4537 | 0.5973 |
| learned_weighted_bank | 0.5426 | 0.4395 | 0.5904 |
| random_row_zscore | 0.4821 | 0.3443 | 0.5394 |

Mean recall by pool:

| strategy | pool 25 | pool 50 | pool 100 | pool 250 |
| --- | --- | --- | --- | --- |
| farthest_row_zscore | 0.7102 | 0.8260 | 0.9051 | 0.9659 |
| learned_top256 | 0.7753 | 0.8880 | 0.9544 | 0.9902 |
| learned_weighted_bank | 0.7692 | 0.8906 | 0.9592 | 0.9936 |
| random_row_zscore | 0.6803 | 0.7960 | 0.8809 | 0.9517 |

## Finding

`learned_top256` beats `farthest_row_zscore` on every model at pool 250, so trained anchor selection is a useful direct routing upgrade. The gain is largest on the previous weak case, `paraphrase-albert-small-v2`, which moves from 0.8903 to 0.9699 at pool 250. The relation-only top-10 check still fails as a complete replacement: pool-10 mean recall is 0.5511, so the right design remains learned relative routing plus raw reranking.

## Artifacts

- CSV: `experiments/learned_anchor_selection/learned_anchor_selection.csv`
- Plot: `experiments/learned_anchor_selection/learned_anchor_selection_pool250.png`
