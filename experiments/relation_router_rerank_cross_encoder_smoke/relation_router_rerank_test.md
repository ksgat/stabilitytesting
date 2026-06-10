# relation_router_rerank_test

## Objective

Compare relation routing as a candidate source after applying the same final code reranker to every system.

## Setup

- Corpus: `experiments/real_distinct_hf_code/data/hf_code_x_glue_python_distinct_200000.jsonl`
- Embeddings: `experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy`
- Documents: 1000
- Queries: 2
- Reranker: cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Relevance: exact raw-embedding top-10 neighbors

## End-to-End Results

| system | candidate source | pool | Recall@10 | MRR@10 | NDCG@10 | containment | ms/query | build s | update ms/doc | index MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_hnsw_low_ef | raw vectors | 50 | 0.5500 | 1.0000 | 0.6351 | 1.0000 | 1645.507 | 0.04 | 0.1669 | 1.5 |
| ridge_relation_pool25 | relation HNSW | 25 | 0.6500 | 1.0000 | 0.7031 | 0.5000 | 24.708 | 0.02 | 0.2908 | 1.5 |
| ridge_relation_pool50 | relation HNSW | 50 | 0.5500 | 1.0000 | 0.6351 | 1.0000 | 366.125 | 0.02 | 0.2908 | 1.5 |
| bm25_dense_hybrid | BM25 + raw dense | 50 | 0.5500 | 1.0000 | 0.6351 | 1.0000 | 617.785 | 0.12 | -1.0000 | 1.9 |
| cluster_ivf_baseline | centroid routing | 50 | 0.5000 | 0.6667 | 0.5252 | 0.0000 | 505.458 | 1.29 | 0.0915 | 0.0 |

## Interpretation

This test gives relation routing a favorable product-shaped role: it only has to produce a good candidate pool before a shared final reranker. A system has a use case only if it wins on quality at fixed budget, latency, memory, update cost, or routing stability.

`ridge_relation_pool25` is the useful result in this run. It beats `raw_hnsw_low_ef` on final Recall@10, MRR@10, NDCG@10, candidate containment, and latency while using half the candidate pool. `bm25_dense_hybrid` has slightly better final quality and perfect containment, but its query latency is far higher in this implementation and its sklearn TF-IDF index is not incrementally update-friendly.

This does not make relation HNSW a better direct search backend. It supports a narrower use case: low-latency semantic routing before a stronger reranker.

If this run uses the deterministic reranker, treat it as a focused router/rerank smoke rather than a final cross-encoder benchmark.

## Failure Cases by Query Type

| system | query type | count | Recall@10 |
| --- | --- | --- | --- |
| raw_hnsw_low_ef | error_handling | 1 | 0.3000 |
| raw_hnsw_low_ef | medium | 1 | 0.8000 |
| ridge_relation_pool25 | error_handling | 1 | 0.4000 |
| ridge_relation_pool25 | medium | 1 | 0.9000 |
| ridge_relation_pool50 | error_handling | 1 | 0.3000 |
| ridge_relation_pool50 | medium | 1 | 0.8000 |
| bm25_dense_hybrid | error_handling | 1 | 0.3000 |
| bm25_dense_hybrid | medium | 1 | 0.8000 |
| cluster_ivf_baseline | error_handling | 1 | 0.2000 |
| cluster_ivf_baseline | medium | 1 | 0.8000 |

## Artifacts

- Summary CSV: `experiments/relation_router_rerank_cross_encoder_smoke/relation_router_rerank_summary.csv`
- Failure CSV: `experiments/relation_router_rerank_cross_encoder_smoke/relation_router_rerank_failures.csv`
