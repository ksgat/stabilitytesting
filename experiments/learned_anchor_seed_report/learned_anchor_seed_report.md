# Learned Anchor Seed Generalization

Pool evaluated: 250

## Strategy Summary

| strategy | mean | std | min | max | rows |
| --- | --- | --- | --- | --- | --- |
| farthest_row_zscore | 0.9673 | 0.0353 | 0.8857 | 0.9940 | 18 |
| learned_top256 | 0.9901 | 0.0091 | 0.9698 | 0.9991 | 18 |
| learned_weighted_bank | 0.9935 | 0.0035 | 0.9845 | 0.9980 | 18 |
| random_row_zscore | 0.9541 | 0.0428 | 0.8574 | 0.9889 | 18 |

## Learned-vs-Farthest by Model

| model | farthest mean | learned mean | delta mean | delta min | delta max |
| --- | --- | --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 | 0.9744 | 0.9903 | +0.0160 | +0.0134 | +0.0178 |
| intfloat/e5-small-v2 | 0.9833 | 0.9911 | +0.0078 | +0.0050 | +0.0114 |
| sentence-transformers/all-MiniLM-L12-v2 | 0.9748 | 0.9944 | +0.0197 | +0.0174 | +0.0220 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.9886 | 0.9957 | +0.0071 | +0.0061 | +0.0081 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 0.9931 | 0.9983 | +0.0052 | +0.0044 | +0.0067 |
| sentence-transformers/paraphrase-albert-small-v2 | 0.8900 | 0.9708 | +0.0808 | +0.0787 | +0.0840 |

## Finding

Across the supplied seeds, learned anchor selection should be considered robust only if its delta over farthest anchors is positive for every model and the variance is small relative to the gain.
