# relation_router_rerank_test

## Objective

Compare relation routing as a candidate source after applying the same final code reranker to every system.

## Setup

- Corpus: `experiments/real_distinct_hf_code/data/hf_code_x_glue_python_distinct_200000.jsonl`
- Embeddings: `experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy`
- Documents: 3000
- Queries: 20
- Reranker: deterministic code reranker = raw dense cosine + token-overlap score
- Relevance: exact raw-embedding top-10 neighbors

## End-to-End Results

| system | candidate source | pool | Recall@10 | MRR@10 | NDCG@10 | containment | ms/query | build s | update ms/doc | index MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_hnsw_low_ef | raw vectors | 50 | 0.8700 | 1.0000 | 0.9120 | 1.0000 | 0.775 | 0.15 | 0.1514 | 4.6 |
| ridge_relation_pool25 | relation HNSW | 25 | 0.8750 | 1.0000 | 0.9153 | 0.9500 | 0.534 | 0.13 | 0.1551 | 4.6 |
| ridge_relation_pool50 | relation HNSW | 50 | 0.8700 | 1.0000 | 0.9120 | 1.0000 | 0.756 | 0.13 | 0.1551 | 4.6 |
| bm25_dense_hybrid | BM25 + raw dense | 50 | 0.8700 | 1.0000 | 0.9120 | 1.0000 | 2.413 | 0.40 | -1.0000 | 5.7 |
| cluster_ivf_baseline | centroid routing | 50 | 0.7400 | 1.0000 | 0.8163 | 0.3500 | 0.641 | 3.20 | 0.0134 | 0.1 |

## Interpretation

This test gives relation routing a favorable product-shaped role: it only has to produce a good candidate pool before a shared final reranker. A system has a use case only if it wins on quality at fixed budget, latency, memory, update cost, or routing stability.

## Failure Cases by Query Type

| system | query type | count | Recall@10 |
| --- | --- | --- | --- |
| raw_hnsw_low_ef | dunder | 3 | 0.8667 |
| raw_hnsw_low_ef | error_handling | 5 | 0.8800 |
| raw_hnsw_low_ef | io | 2 | 0.9000 |
| raw_hnsw_low_ef | long | 3 | 0.8667 |
| raw_hnsw_low_ef | medium | 5 | 0.8400 |
| raw_hnsw_low_ef | test_assert | 2 | 0.9000 |
| ridge_relation_pool25 | dunder | 3 | 0.8667 |
| ridge_relation_pool25 | error_handling | 5 | 0.8800 |
| ridge_relation_pool25 | io | 2 | 0.9000 |
| ridge_relation_pool25 | long | 3 | 0.8667 |
| ridge_relation_pool25 | medium | 5 | 0.8600 |
| ridge_relation_pool25 | test_assert | 2 | 0.9000 |
| ridge_relation_pool50 | dunder | 3 | 0.8667 |
| ridge_relation_pool50 | error_handling | 5 | 0.8800 |
| ridge_relation_pool50 | io | 2 | 0.9000 |
| ridge_relation_pool50 | long | 3 | 0.8667 |
| ridge_relation_pool50 | medium | 5 | 0.8400 |
| ridge_relation_pool50 | test_assert | 2 | 0.9000 |
| bm25_dense_hybrid | dunder | 3 | 0.8667 |
| bm25_dense_hybrid | error_handling | 5 | 0.8800 |
| bm25_dense_hybrid | io | 2 | 0.9000 |
| bm25_dense_hybrid | long | 3 | 0.8667 |
| bm25_dense_hybrid | medium | 5 | 0.8400 |
| bm25_dense_hybrid | test_assert | 2 | 0.9000 |
| cluster_ivf_baseline | dunder | 3 | 0.8000 |
| cluster_ivf_baseline | error_handling | 5 | 0.6400 |
| cluster_ivf_baseline | io | 2 | 0.8000 |
| cluster_ivf_baseline | long | 3 | 0.8333 |
| cluster_ivf_baseline | medium | 5 | 0.7400 |
| cluster_ivf_baseline | test_assert | 2 | 0.7000 |

## Artifacts

- Summary CSV: `experiments/relation_router_rerank_smoke/relation_router_rerank_summary.csv`
- Failure CSV: `experiments/relation_router_rerank_smoke/relation_router_rerank_failures.csv`
