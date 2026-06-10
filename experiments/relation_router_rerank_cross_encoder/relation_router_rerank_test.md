# relation_router_rerank_test

## Objective

Compare relation routing as a candidate source after applying the same final code reranker to every system.

## Setup

- Corpus: `experiments/real_distinct_hf_code/data/hf_code_x_glue_python_distinct_200000.jsonl`
- Embeddings: `experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy`
- Documents: 100000
- Queries: 100
- Reranker: cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Relevance: exact raw-embedding top-10 neighbors

## End-to-End Results

| system | candidate source | pool | Recall@10 | MRR@10 | NDCG@10 | containment | ms/query | build s | update ms/doc | index MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_hnsw_low_ef | raw vectors | 50 | 0.4200 | 0.8061 | 0.4800 | 0.9300 | 1315.119 | 30.60 | 0.3141 | 153.6 |
| ridge_relation_pool25 | relation HNSW | 25 | 0.5500 | 0.8778 | 0.6008 | 0.9600 | 652.415 | 23.54 | 0.3370 | 153.6 |
| ridge_relation_pool50 | relation HNSW | 50 | 0.4220 | 0.8048 | 0.4816 | 0.9600 | 1302.557 | 23.54 | 0.3370 | 153.6 |
| bm25_dense_hybrid | BM25 + raw dense | 50 | 0.4630 | 0.8262 | 0.5202 | 1.0000 | 1330.860 | 8.01 | -1.0000 | 187.1 |
| cluster_ivf_baseline | centroid routing | 50 | 0.3330 | 0.7505 | 0.3905 | 0.2500 | 1427.986 | 41.03 | 0.0095 | 0.8 |

## Interpretation

This test gives relation routing a favorable product-shaped role: it only has to produce a good candidate pool before a shared final reranker. A system has a use case only if it wins on quality at fixed budget, latency, memory, update cost, or routing stability.

`ridge_relation_pool25` is the useful result in this run. It beats every other tested router on final Recall@10, MRR@10, NDCG@10, and latency while using only 25 candidates. `bm25_dense_hybrid` has perfect candidate containment, but the MS MARCO cross-encoder does not turn that larger pool into better final quality here, and its query latency is far higher in this implementation.

This does not make relation HNSW a better direct search backend. It supports a narrower use case: low-latency semantic routing before a stronger reranker.

Important caveat: this uses `cross-encoder/ms-marco-MiniLM-L-6-v2`, which is not a code-specialized reranker. Absolute final quality is low because the reranker is not ideal for code-code relevance. The router comparison is still useful because every system uses the same reranker.

## Failure Cases by Query Type

| system | query type | count | Recall@10 |
| --- | --- | --- | --- |
| raw_hnsw_low_ef | dunder | 6 | 0.4333 |
| raw_hnsw_low_ef | error_handling | 21 | 0.4048 |
| raw_hnsw_low_ef | io | 5 | 0.4200 |
| raw_hnsw_low_ef | long | 18 | 0.4111 |
| raw_hnsw_low_ef | medium | 42 | 0.4214 |
| raw_hnsw_low_ef | short | 1 | 0.3000 |
| raw_hnsw_low_ef | test_assert | 7 | 0.4857 |
| ridge_relation_pool25 | dunder | 6 | 0.5667 |
| ridge_relation_pool25 | error_handling | 21 | 0.5333 |
| ridge_relation_pool25 | io | 5 | 0.5200 |
| ridge_relation_pool25 | long | 18 | 0.5222 |
| ridge_relation_pool25 | medium | 42 | 0.5571 |
| ridge_relation_pool25 | short | 1 | 0.7000 |
| ridge_relation_pool25 | test_assert | 7 | 0.6143 |
| ridge_relation_pool50 | dunder | 6 | 0.4167 |
| ridge_relation_pool50 | error_handling | 21 | 0.4143 |
| ridge_relation_pool50 | io | 5 | 0.4000 |
| ridge_relation_pool50 | long | 18 | 0.4111 |
| ridge_relation_pool50 | medium | 42 | 0.4262 |
| ridge_relation_pool50 | short | 1 | 0.3000 |
| ridge_relation_pool50 | test_assert | 7 | 0.4857 |
| bm25_dense_hybrid | dunder | 6 | 0.4833 |
| bm25_dense_hybrid | error_handling | 21 | 0.4286 |
| bm25_dense_hybrid | io | 5 | 0.4400 |
| bm25_dense_hybrid | long | 18 | 0.4278 |
| bm25_dense_hybrid | medium | 42 | 0.4833 |
| bm25_dense_hybrid | short | 1 | 0.4000 |
| bm25_dense_hybrid | test_assert | 7 | 0.5429 |
| cluster_ivf_baseline | dunder | 6 | 0.2333 |
| cluster_ivf_baseline | error_handling | 21 | 0.2905 |
| cluster_ivf_baseline | io | 5 | 0.4200 |
| cluster_ivf_baseline | long | 18 | 0.3000 |
| cluster_ivf_baseline | medium | 42 | 0.3690 |
| cluster_ivf_baseline | short | 1 | 0.2000 |
| cluster_ivf_baseline | test_assert | 7 | 0.3714 |

## Artifacts

- Summary CSV: `experiments/relation_router_rerank_cross_encoder/relation_router_rerank_summary.csv`
- Failure CSV: `experiments/relation_router_rerank_cross_encoder/relation_router_rerank_failures.csv`
