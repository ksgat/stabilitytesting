# Relation Index Scale Benchmark

## Scope

- Base embedding file: `experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy`
- Sizes: 1000, 3000, 10000, 100000, 200000
- Anchors: 1024
- Ridge: 0.03
- Sizes above the base embedding count use noisy resampling and are mechanics-only. All listed sizes here are real subsets because they are <= the 200k base embedding count.

## Real-Subset Relation-Only Quality at Pool 10

| docs | method | recall | all top-k | ms/query |
| --- | --- | --- | --- | --- |
| 1000 | cosine_relation | 0.6685 | 0.0100 | 0.045 |
| 1000 | ridge_bilinear | 0.8865 | 0.2600 | 0.043 |
| 3000 | cosine_relation | 0.6172 | 0.0240 | 0.074 |
| 3000 | ridge_bilinear | 0.9232 | 0.4060 | 0.054 |
| 10000 | cosine_relation | 0.6148 | 0.0100 | 0.265 |
| 10000 | ridge_bilinear | 0.9536 | 0.5780 | 0.186 |
| 100000 | cosine_relation | 0.6038 | 0.0120 | 1.554 |
| 100000 | ridge_bilinear | 0.9638 | 0.6760 | 0.930 |
| 200000 | cosine_relation | 0.5636 | 0.0060 | 3.219 |
| 200000 | ridge_bilinear | 0.9614 | 0.6440 | 1.984 |

## HNSW Mechanics

| docs | mode | method | build s | ms/query | vector MB |
| --- | --- | --- | --- | --- | --- |
| 1000 | real_subset | ridge_bilinear | 0.05 | 0.031 | 1.5 |
| 3000 | real_subset | ridge_bilinear | 0.18 | 0.055 | 4.6 |
| 10000 | real_subset | ridge_bilinear | 1.05 | 0.100 | 15.4 |
| 100000 | real_subset | ridge_bilinear | 21.89 | 0.184 | 153.6 |
| 200000 | real_subset | ridge_bilinear | 51.83 | 0.201 | 307.2 |

## Interpretation

Use real-subset rows for quality. Use synthetic rows only to estimate mechanics: vector memory, HNSW build time, and query latency. Here, both 100k and 200k are real quality rows from distinct dataset examples.

## Artifacts

- CSV: `experiments/real_distinct_hf_code/relation_index_scale/relation_index_scale.csv`
- Plot: `experiments/real_distinct_hf_code/relation_index_scale/relation_index_scale_latency.png`
