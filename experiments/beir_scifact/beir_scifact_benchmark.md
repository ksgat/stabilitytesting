# BEIR SciFact Benchmark

- Corpus documents: 5183
- Test queries with qrels: 300
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Anchor count: 256

| method | pool | recall@10 | mrr@10 | ndcg@10 | queries |
| --- | --- | --- | --- | --- | --- |
| raw_exact | 10 | 0.7817 | 0.5840 | 0.6292 | 300 |
| raw_hnsw | 10 | 0.7817 | 0.5840 | 0.6292 | 300 |
| larp_relation_only | 10 | 0.6314 | 0.4356 | 0.4774 | 300 |
| larp_hnsw_rerank | 100 | 0.7510 | 0.5750 | 0.6140 | 300 |
| larp_hnsw_rerank | 250 | 0.7667 | 0.5813 | 0.6228 | 300 |
| larp_hnsw_rerank | 500 | 0.7750 | 0.5829 | 0.6269 | 300 |

## Finding

This is a real labeled retrieval benchmark, not a raw-neighbor proxy. If LARP reranking matches raw HNSW here, the routing layer preserves task relevance well enough for this small benchmark. If relation-only lags, reranking remains necessary.

## Artifacts

- CSV: `experiments/beir_scifact/beir_scifact_benchmark.csv`
- Doc embeddings: `experiments/beir_scifact/scifact_doc_embeddings.npy`
- Query embeddings: `experiments/beir_scifact/scifact_query_embeddings.npy`
