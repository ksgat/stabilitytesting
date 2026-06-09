# Anchor Ablation

- Docs: 10000
- Anchors: 256
- Queries: 1000
- Raw target: top-10

## Top Results At Pool 250

| Anchor Strategy | Transform | Mean Recall | All Top-k | Any Hit |
|---|---|---:|---:|---:|
| farthest_random | row_zscore | 0.9850 | 0.8890 | 1.0000 |
| kmeans_boundary | row_zscore | 0.9835 | 0.8780 | 1.0000 |
| farthest_random | raw | 0.9824 | 0.8710 | 1.0000 |
| kmeans_boundary | raw | 0.9802 | 0.8580 | 1.0000 |
| multi_scale | row_zscore | 0.9791 | 0.8560 | 1.0000 |
| density_sparse | row_zscore | 0.9790 | 0.8480 | 1.0000 |
| farthest_random | rank | 0.9776 | 0.8420 | 1.0000 |
| density_sparse | raw | 0.9770 | 0.8400 | 1.0000 |
| kmeans_boundary | rank | 0.9769 | 0.8450 | 1.0000 |
| multi_scale | raw | 0.9742 | 0.8240 | 1.0000 |
| density_sparse | rank | 0.9721 | 0.8070 | 1.0000 |
| multi_scale | rank | 0.9715 | 0.8020 | 1.0000 |
| random | row_zscore | 0.9573 | 0.7400 | 1.0000 |
| random | raw | 0.9522 | 0.7150 | 1.0000 |
| random | rank | 0.9472 | 0.6930 | 1.0000 |
