# Relation Index Scale Benchmark

## Scope

- Base embedding file: `experiments/real_100k_chunks/embeddings/code_chunks_100k_minilm.npy`
- Sizes: 1000, 3000, 10000, 100000
- Anchors: 1024
- Ridge: 0.03
- Synthetic sizes above the base embedding count use noisy resampling and are mechanics-only.

## Real-Subset Relation-Only Quality at Pool 10

| docs | method | recall | all top-k | ms/query |
| --- | --- | --- | --- | --- |
| 1000 | cosine_relation | 0.7980 | 0.1750 | 0.025 |
| 1000 | ridge_bilinear | 0.9165 | 0.4750 | 0.022 |
| 3000 | cosine_relation | 0.7794 | 0.1140 | 0.055 |
| 3000 | ridge_bilinear | 0.9472 | 0.5720 | 0.036 |
| 10000 | cosine_relation | 0.7910 | 0.1520 | 0.217 |
| 10000 | ridge_bilinear | 0.9658 | 0.7080 | 0.131 |
| 100000 | cosine_relation | 0.8006 | 0.1420 | 1.644 |
| 100000 | ridge_bilinear | 0.9804 | 0.8100 | 0.881 |

## HNSW Mechanics

| docs | mode | method | build s | ms/query | vector MB |
| --- | --- | --- | --- | --- | --- |
| 1000 | real_subset | ridge_bilinear | 0.05 | 0.019 | 1.5 |
| 3000 | real_subset | ridge_bilinear | 0.12 | 0.030 | 4.6 |
| 10000 | real_subset | ridge_bilinear | 0.62 | 0.045 | 15.4 |
| 100000 | real_subset | ridge_bilinear | 10.87 | 0.115 | 153.6 |

## Interpretation

Use real-subset rows for quality. Use synthetic rows only to estimate mechanics: vector memory, HNSW build time, and query latency. A real 100k/200k quality test requires embedding that many real chunks.

## Artifacts

- CSV: `experiments/real_100k_chunks/relation_index_scale/relation_index_scale.csv`
- Plot: `experiments/real_100k_chunks/relation_index_scale/relation_index_scale_latency.png`
