# Relation Index Scale Benchmark

## Scope

- Base embedding file: `experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy`
- Sizes: 1000, 3000, 10000, 100000, 200000
- Anchors: 1024
- Ridge: 0.03
- Synthetic sizes above the base embedding count use noisy resampling and are mechanics-only.

## Real-Subset Relation-Only Quality at Pool 10

| docs | method | recall | all top-k | ms/query |
| --- | --- | --- | --- | --- |
| 1000 | cosine_relation | 0.5945 | 0.0050 | 0.161 |
| 1000 | ridge_bilinear | 0.8670 | 0.1850 | 0.085 |
| 3000 | cosine_relation | 0.6150 | 0.0080 | 0.388 |
| 3000 | ridge_bilinear | 0.9070 | 0.3040 | 0.404 |
| 10000 | cosine_relation | 0.6608 | 0.0240 | 1.062 |
| 10000 | ridge_bilinear | 0.9224 | 0.4160 | 0.592 |

## HNSW Mechanics

| docs | mode | method | build s | ms/query | vector MB |
| --- | --- | --- | --- | --- | --- |
| 1000 | real_subset | ridge_bilinear | 0.06 | 0.178 | 1.5 |
| 3000 | real_subset | ridge_bilinear | 0.17 | 0.542 | 4.6 |
| 10000 | real_subset | ridge_bilinear | 1.11 | 0.647 | 15.4 |
| 100000 | synthetic_expanded | ridge_bilinear | 20.51 | 0.570 | 153.6 |
| 200000 | synthetic_expanded | ridge_bilinear | 42.15 | 0.505 | 307.2 |

## Interpretation

Use real-subset rows for quality. Use synthetic rows only to estimate mechanics: vector memory, HNSW build time, and query latency. A real 100k/200k quality test requires embedding that many real chunks.

## Artifacts

- CSV: `experiments/relation_index_scale/relation_index_scale.csv`
- Plot: `experiments/relation_index_scale/relation_index_scale_latency.png`
