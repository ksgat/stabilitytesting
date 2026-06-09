# Raw Vector Search Baseline

## Scope

- Embeddings: `experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy`
- Sizes: 100000, 200000
- Queries per size: 500
- HNSW: M=32, ef_construction=200, ef_search=128
- Relation metric: 1024 anchors, ridge=0.03

Truth is exact cosine top-10 over raw embeddings. `raw_exact` is therefore recall 1.0 by definition and reports brute-force exact latency.

## Pool-10 Comparison

| docs | method | recall | all top-k | build s | prep s | ms/query | vector MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | raw_exact | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.469 | 153.6 |
| 100000 | raw_hnsw | 0.9978 | 0.9840 | 29.44 | 0.00 | 0.365 | 153.6 |
| 100000 | ridge_relation_hnsw | 0.9550 | 0.5940 | 23.58 | 22.30 | 0.196 | 153.6 |
| 200000 | raw_exact | 1.0000 | 1.0000 | 0.00 | 0.00 | 2.858 | 307.2 |
| 200000 | raw_hnsw | 0.9924 | 0.9560 | 56.13 | 0.00 | 0.224 | 307.2 |
| 200000 | ridge_relation_hnsw | 0.9584 | 0.6260 | 57.10 | 28.51 | 0.226 | 307.2 |

## Pool-25 Candidate Comparison

| docs | method | recall of raw top-10 | all top-10 contained |
| --- | --- | --- | --- |
| 100000 | raw_hnsw | 0.9978 | 0.9840 |
| 100000 | ridge_relation_hnsw | 0.9970 | 0.9740 |
| 200000 | raw_hnsw | 0.9924 | 0.9560 |
| 200000 | ridge_relation_hnsw | 0.9940 | 0.9540 |

## Interpretation

Raw HNSW is the direct vector-search baseline. Ridge-bilinear relation HNSW must beat this on latency, memory, freshness, or quality to justify itself as a primary search backend.

On this run, raw HNSW wins as the primary top-10 backend. Ridge-relation HNSW is competitive only as a small candidate-pool generator: at pool 25 it nearly matches raw HNSW containment, but at direct top-10 it loses too much exact-neighbor recall.

## Artifacts

- CSV: `experiments/real_distinct_hf_code/raw_vector_baseline/raw_vector_search_baseline.csv`
