# Anchor Ablation

- Docs: 10000
- Anchors: 256
- Queries: 1000
- Raw target: top-10

## Top Results At Pool 500

| Anchor Strategy | Transform | Mean Recall | All Top-k | Any Hit |
|---|---|---:|---:|---:|
| farthest_random | row_zscore | 0.9962 | 0.9670 | 1.0000 |
| farthest_random | raw | 0.9947 | 0.9560 | 1.0000 |
| farthest_random | rank | 0.9945 | 0.9530 | 1.0000 |
| density_sparse | row_zscore | 0.9931 | 0.9450 | 1.0000 |
| kmeans_boundary | row_zscore | 0.9926 | 0.9410 | 1.0000 |
| farthest_centroid | row_zscore | 0.9920 | 0.9340 | 1.0000 |
| kmeans_boundary | raw | 0.9910 | 0.9300 | 1.0000 |
| density_sparse | raw | 0.9910 | 0.9300 | 1.0000 |
| density_sparse | rank | 0.9906 | 0.9230 | 1.0000 |
| kmeans_boundary | rank | 0.9905 | 0.9260 | 1.0000 |
| multi_scale | row_zscore | 0.9904 | 0.9230 | 1.0000 |
| farthest_centroid | raw | 0.9904 | 0.9200 | 1.0000 |
| multi_scale | raw | 0.9882 | 0.9080 | 1.0000 |
| multi_scale | rank | 0.9881 | 0.9070 | 1.0000 |
| farthest_centroid | rank | 0.9876 | 0.9010 | 1.0000 |
| pca_extremes | row_zscore | 0.9857 | 0.8910 | 1.0000 |
| random | row_zscore | 0.9834 | 0.8800 | 1.0000 |
| density_sparse | softmax_0.1 | 0.9819 | 0.8580 | 1.0000 |
| pca_extremes | raw | 0.9809 | 0.8630 | 1.0000 |
| language_random | row_zscore | 0.9796 | 0.8600 | 1.0000 |
