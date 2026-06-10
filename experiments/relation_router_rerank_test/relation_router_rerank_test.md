# relation_router_rerank_test

## Objective

Compare relation routing as a candidate source after applying the same final code reranker to every system.

## Setup

- Corpus: `experiments/real_distinct_hf_code/data/hf_code_x_glue_python_distinct_200000.jsonl`
- Embeddings: `experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy`
- Documents: 100000
- Queries: 100
- Reranker: deterministic code reranker = raw dense cosine + token-overlap score
- Relevance: exact raw-embedding top-10 neighbors

## End-to-End Results

| system | candidate source | pool | Recall@10 | MRR@10 | NDCG@10 | containment | ms/query | build s | update ms/doc | index MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_hnsw_low_ef | raw vectors | 50 | 0.7720 | 0.9900 | 0.8311 | 0.9100 | 0.868 | 35.59 | 0.4097 | 153.6 |
| ridge_relation_pool25 | relation HNSW | 25 | 0.7970 | 0.9950 | 0.8508 | 0.9600 | 0.686 | 34.69 | 0.4218 | 153.6 |
| ridge_relation_pool50 | relation HNSW | 50 | 0.7750 | 0.9950 | 0.8349 | 0.9600 | 0.937 | 34.69 | 0.4218 | 153.6 |
| bm25_dense_hybrid | BM25 + raw dense | 50 | 0.8030 | 0.9900 | 0.8530 | 1.0000 | 63.086 | 14.86 | -1.0000 | 187.1 |
| cluster_ivf_baseline | centroid routing | 50 | 0.6240 | 0.9750 | 0.7148 | 0.2500 | 1.156 | 54.39 | 0.0100 | 0.8 |

## Interpretation

This test gives relation routing a favorable product-shaped role: it only has to produce a good candidate pool before a shared final reranker. A system has a use case only if it wins on quality at fixed budget, latency, memory, update cost, or routing stability.

`ridge_relation_pool25` is the useful result in this run. It beats `raw_hnsw_low_ef` on final Recall@10, MRR@10, NDCG@10, candidate containment, and latency while using half the candidate pool. `bm25_dense_hybrid` has slightly better final quality and perfect containment, but its query latency is far higher in this implementation and its sklearn TF-IDF index is not incrementally update-friendly.

This does not make relation HNSW a better direct search backend. It supports a narrower use case: low-latency semantic routing before a stronger reranker.

Limitation: this run used a deterministic code reranker because the transformer cross-encoder download/load timed out in the environment. Treat it as a focused router/rerank smoke, not a final cross-encoder benchmark.

## Failure Cases by Query Type

| system | query type | count | Recall@10 |
| --- | --- | --- | --- |
| raw_hnsw_low_ef | dunder | 6 | 0.8167 |
| raw_hnsw_low_ef | error_handling | 21 | 0.7476 |
| raw_hnsw_low_ef | io | 5 | 0.7600 |
| raw_hnsw_low_ef | long | 18 | 0.8056 |
| raw_hnsw_low_ef | medium | 42 | 0.7524 |
| raw_hnsw_low_ef | short | 1 | 0.8000 |
| raw_hnsw_low_ef | test_assert | 7 | 0.8429 |
| ridge_relation_pool25 | dunder | 6 | 0.8333 |
| ridge_relation_pool25 | error_handling | 21 | 0.7905 |
| ridge_relation_pool25 | io | 5 | 0.7600 |
| ridge_relation_pool25 | long | 18 | 0.8167 |
| ridge_relation_pool25 | medium | 42 | 0.7786 |
| ridge_relation_pool25 | short | 1 | 0.9000 |
| ridge_relation_pool25 | test_assert | 7 | 0.8571 |
| ridge_relation_pool50 | dunder | 6 | 0.8167 |
| ridge_relation_pool50 | error_handling | 21 | 0.7571 |
| ridge_relation_pool50 | io | 5 | 0.7400 |
| ridge_relation_pool50 | long | 18 | 0.8056 |
| ridge_relation_pool50 | medium | 42 | 0.7595 |
| ridge_relation_pool50 | short | 1 | 0.8000 |
| ridge_relation_pool50 | test_assert | 7 | 0.8286 |
| bm25_dense_hybrid | dunder | 6 | 0.8667 |
| bm25_dense_hybrid | error_handling | 21 | 0.7857 |
| bm25_dense_hybrid | io | 5 | 0.7600 |
| bm25_dense_hybrid | long | 18 | 0.8278 |
| bm25_dense_hybrid | medium | 42 | 0.7857 |
| bm25_dense_hybrid | short | 1 | 0.9000 |
| bm25_dense_hybrid | test_assert | 7 | 0.8571 |
| cluster_ivf_baseline | dunder | 6 | 0.5000 |
| cluster_ivf_baseline | error_handling | 21 | 0.5857 |
| cluster_ivf_baseline | io | 5 | 0.6400 |
| cluster_ivf_baseline | long | 18 | 0.5722 |
| cluster_ivf_baseline | medium | 42 | 0.6690 |
| cluster_ivf_baseline | short | 1 | 0.8000 |
| cluster_ivf_baseline | test_assert | 7 | 0.6714 |

## Artifacts

- Summary CSV: `experiments/relation_router_rerank_test/relation_router_rerank_summary.csv`
- Failure CSV: `experiments/relation_router_rerank_test/relation_router_rerank_failures.csv`
