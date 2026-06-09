# LARP HNSW System Benchmark

## Scope

- Embedding file: `experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy`
- Corpus path: `experiments/tenk_minilm_candidate/data/processed/hf_google_code_x_glue_ct_code_to_text_n10000_seed7.jsonl`
- Build docs: 8000
- Final docs after inserts: 10000
- Query count per stage: 300
- Top-k: 10
- LARP candidate pools: 100, 250, 500

## What This Covers

- Raw-HNSW baseline against exact raw cosine top-k.
- LARP-HNSW over relative signatures with raw reranking.
- Incremental inserts without recomputing existing signatures.
- Drift diagnostics over new batches.
- Multi-generation search simulation.
- Relation-only top-k failure mode.
- A weak label-based relevance proxy from corpus metadata.
- Signature compression/memory estimates.

## Final Dynamic Insert Stage (10000 docs)

| method | pool | recall | all top-k | ms/query | label precision |
| --- | --- | --- | --- | --- | --- |
| raw_hnsw | 10 | 1.0000 | 1.0000 | 1.195 | 0.8090 |
| larp_hnsw_rerank | 100 | 0.9513 | 0.7267 | 1.735 | 0.8183 |
| larp_hnsw_rerank | 250 | 0.9850 | 0.8933 | 2.466 | 0.8133 |
| larp_hnsw_rerank | 500 | 0.9940 | 0.9500 | 4.077 | 0.8110 |

## Relation-Only Check

| method | pool | recall | all top-k | ms/query | label precision |
| --- | --- | --- | --- | --- | --- |
| larp_hnsw_relation_only | 10 | 0.6383 | 0.0067 | 1.340 | 0.8243 |

## Multi-Generation Simulation

| method | pool | recall | all top-k | ms/query | label precision |
| --- | --- | --- | --- | --- | --- |
| multi_generation_larp | 500 | 0.9963 | 0.9667 | 6.092 | 0.7963 |

## Drift Diagnostics

| stage | doc_count | top-anchor sim | entropy | anchor gini |
| --- | --- | --- | --- | --- |
| build | 8000 | 0.7682 | 5.5439 | 0.5189 |
| insert_batch_1 | 500 | 0.7553 | 5.5439 | 0.6383 |
| insert_batch_2 | 500 | 0.7570 | 5.5439 | 0.6211 |
| insert_batch_3 | 1000 | 0.7588 | 5.5439 | 0.5714 |

## Compression Estimate

| representation | MB |
| --- | --- |
| raw float32 | 15.360 |
| signature float32 | 10.240 |
| signature float16 | 5.120 |
| signature int8 | 2.560 |

## Interpretation

The decisive comparison is raw-HNSW versus LARP-HNSW after inserts. LARP only has a practical advantage if it reaches comparable recall with lower latency, smaller routing memory, better insert behavior, or better degradation properties under append-heavy updates. Relation-only retrieval is reported separately because it is not the same claim as high-recall routing.

The label precision column is a weak metadata proxy, not a human relevance benchmark. It is included only to avoid relying exclusively on raw embedding top-k as the evaluation target.

## Artifacts

- CSV: `experiments/larp_hnsw_system/larp_hnsw_system_benchmark.csv`
- Drift CSV: `experiments/larp_hnsw_system/larp_hnsw_drift.csv`
- Plot: `experiments/larp_hnsw_system/larp_hnsw_dynamic_recall.png`
