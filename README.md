# Relative Signature Index Experiments

## Result Up Front

This project tested whether fixed anchor-relative signatures could become a superior semantic search index: stable under new insertions, searchable directly as relation vectors, and competitive with or better than ordinary raw-vector HNSW.

The answer is mostly no.

The useful result is narrower: ridge-bilinear projected relation signatures are a strong small-pool routing layer, but they are not a better direct top-k backend than raw vector search.

On 200k distinct real CodeXGLUE Python rows, raw HNSW wins direct top-10 search:

| Docs | Method | Recall of Exact Raw Top-10 | All Top-10 Contained | Build / Prep | ms/query |
|---:|---|---:|---:|---:|---:|
| 100,000 | raw_hnsw | 0.9978 | 0.9840 | 29.44 s | 0.365 |
| 100,000 | ridge_relation_hnsw | 0.9550 | 0.5940 | 23.58 s build + 22.30 s prep | 0.196 |
| 200,000 | raw_hnsw | 0.9924 | 0.9560 | 56.13 s | 0.224 |
| 200,000 | ridge_relation_hnsw | 0.9584 | 0.6260 | 57.10 s build + 28.51 s prep | 0.226 |

At pool 25, ridge-relation HNSW nearly matches raw HNSW as a candidate generator:

| Docs | Method | Recall of Exact Raw Top-10 | All Top-10 Contained |
|---:|---|---:|---:|
| 100,000 | raw_hnsw | 0.9978 | 0.9840 |
| 100,000 | ridge_relation_hnsw | 0.9970 | 0.9740 |
| 200,000 | raw_hnsw | 0.9924 | 0.9560 |
| 200,000 | ridge_relation_hnsw | 0.9940 | 0.9540 |

So the final conclusion is:

> Fixed anchor-relative signatures do not beat raw vector search as a primary top-k index. Ridge-bilinear projection makes them useful as a compact semantic routing/candidate layer, but not as a standalone superior search backend.

## Focused Router/Rerank Result

I ran one more product-shaped test under `relation_router_rerank_test`: every system only had to route candidates, then the same cross-encoder reranker reranked the candidate set. This changes the question from "can relation vectors replace raw vector search?" to "can relation routing win on some operational axis after reranking?"

Setup:

- 100,000 distinct CodeXGLUE Python rows
- 100 held-out queries
- Relevance target: exact raw-embedding top-10 neighbors
- Final reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` for all systems
- Candidate routers: low-ef raw HNSW, ridge-relation HNSW at pool 25/50, BM25+dense hybrid, and centroid/IVF routing

Result:

| System | Candidate Source | Pool | Recall@10 | MRR@10 | NDCG@10 | Candidate Containment | ms/query | Build | Update ms/doc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_hnsw_low_ef | raw vectors | 50 | 0.4200 | 0.8061 | 0.4800 | 0.9300 | 1315.119 | 30.60 s | 0.3141 |
| ridge_relation_pool25 | relation HNSW | 25 | 0.5500 | 0.8778 | 0.6008 | 0.9600 | 652.415 | 23.54 s | 0.3370 |
| ridge_relation_pool50 | relation HNSW | 50 | 0.4220 | 0.8048 | 0.4816 | 0.9600 | 1302.557 | 23.54 s | 0.3370 |
| bm25_dense_hybrid | BM25 + raw dense | 50 | 0.4630 | 0.8262 | 0.5202 | 1.0000 | 1330.860 | 8.01 s | n/a |
| cluster_ivf_baseline | centroid routing | 50 | 0.3330 | 0.7505 | 0.3905 | 0.2500 | 1427.986 | 41.03 s | 0.0095 |

This finds a narrow use case: `ridge_relation_pool25` is the best router in this setup. It beats low-ef raw HNSW, BM25+dense hybrid, and centroid routing on final Recall@10/MRR@10/NDCG@10 while using only 25 candidates. It is also much faster than the 50-candidate cross-encoder systems because it reranks half as many pairs.

Caveat: the cross-encoder is an MS MARCO reranker, not a code-specialized reranker, so absolute final quality is low. The comparison is still useful because every candidate source uses the same final reranker.

This does not overturn the direct-search result. The relation layer is still not a superior standalone top-k backend. The defensible product role is:

> low-latency semantic candidate routing before a stronger reranker.

## What I Thought I Was Doing

The original idea was that embeddings might be converted into stable relative coordinates: instead of storing/searching raw embeddings directly, represent every document by its relations to a fixed anchor set.

The hoped-for product/research claim was:

1. Pick anchors once.
2. Convert every document into a relation signature.
3. Add new documents by embedding only the new item and comparing it to the fixed anchors.
4. Avoid recomputing the full corpus when new data arrives.
5. Search relation signatures directly and recover top-k semantic neighbors.

The strong version was not just "candidate generation." The strong version was relation signature -> top-k search, with a stable incremental index that grows naturally as files, commits, or documents are added.

## What Was Actually Happening

Plain cosine search over relation signatures was not preserving nearest-neighbor order well enough. The relation vector contained useful geometric information, but the naive metric was wrong.

The breakthrough was ridge-bilinear projection. Instead of treating anchor similarities as the final search space, we treated them as coordinates and learned a regularized map back toward raw embedding geometry:

```text
relation signature -> ridge projection -> projected relation vector
```

That made relation-only retrieval look much stronger. On distinct 100k/200k code rows, ridge-relation vectors reached about `0.96` recall of exact raw top-10 at pool 10, and pool 25 recovered almost all exact raw top-10 neighbors.

But the raw-vector baseline changed the interpretation. Raw HNSW already solved the direct search problem better. The ridge-relation system was mostly reconstructing raw embedding geometry with extra machinery, not surpassing it.

## What We Did

The project tested the idea in increasingly less-forgiving settings:

1. Small anchor-relative candidate recall experiments.
2. Multi-model robustness checks across sentence-transformer architectures.
3. Anchor strategy ablations.
4. Learned anchor selection tests.
5. HNSW-backed incremental index prototypes.
6. BEIR SciFact labeled retrieval checks.
7. Ridge/contrastive bilinear metric learning over relation signatures.
8. Synthetic 100k/200k mechanics tests.
9. Overlapping 100k code chunk test, later treated as too saturated.
10. Real distinct 100k/200k CodeXGLUE Python tests from Hugging Face Parquet rows.
11. Raw exact and raw HNSW baselines on the same 100k/200k embeddings.

The final serious dataset was `google/code_x_glue_ct_code_to_text`, Python train split. It used 200,000 distinct code rows deduped by SHA-256 of code text. No overlap chunk expansion and no synthetic resampling were used for the final 100k/200k quality claim.

## End Result

The project answered the research question:

- As a direct top-k vector search replacement: no.
- As a small-pool semantic routing layer: yes, plausible.
- As a superior search product by itself: no.
- As a component inside a hybrid code search system: maybe, but raw HNSW/BM25/code-structure baselines matter more.

The most honest summary:

> Interesting measurable signal, not enough product differentiation.

## Environment

The existing outputs in this checkout were produced with the system Python available on this machine. For reproduction, use a virtual environment.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

GPU is optional. The scripts default to CUDA if PyTorch sees it, otherwise CPU.

## Repo Layout

```text
run_larp_experiment.py          Main experiment runner
larp_hnsw_index.py              HNSW-backed LARP index API and generation facade
scripts/07_signature_index_benchmark.py
                                Relative-index speed benchmark
scripts/10_model_robustness.py
                                Multi-model cached-embedding robustness report
scripts/11_learned_anchor_selection.py
                                Learned anchor selection experiment
scripts/12_larp_hnsw_system_benchmark.py
                                Raw-HNSW vs LARP-HNSW dynamic insert benchmark
scripts/13_learned_anchor_seed_report.py
                                Learned-anchor seed aggregation report
scripts/14_beir_scifact_benchmark.py
                                BEIR SciFact labeled relevance benchmark
scripts/15_bilinear_relation_metric.py
                                Relation-only bilinear metric learning
scripts/16_ridge_bilinear_sweep.py
                                Time-budgeted ridge-bilinear relation-only sweep
scripts/17_scale_relation_index.py
                                Small/100k/200k relation-index scale benchmark
scripts/00_*.py ... 06_*.py     Earlier staged pipeline scripts
results.md                      Consolidated findings
reports/                        Generated per-run markdown reports
outputs/                        Generated baseline artifacts
experiments/tenk_minilm_candidate/
                                Separate 10k candidate-recall run
data/                           Downloaded and processed local data
```

`data/`, embedding arrays, Python caches, and virtual environments are ignored for future runs. Some generated artifacts were already tracked before cleanup; see the cleanup note below.

## Reproduce Key Runs

Small code-model run:

```powershell
python .\run_larp_experiment.py `
  --model-suite code `
  --n-docs 120 `
  --anchor-count 32 `
  --batch-size 8 `
  --max-tokens 160 `
  --run-name code_models_candidate `
  --artifact-root experiments\code_models_candidate
```

General sentence-model suite:

```powershell
python .\run_larp_experiment.py `
  --model-suite general-sentence `
  --n-docs 120 `
  --anchor-count 32 `
  --batch-size 8 `
  --max-tokens 160 `
  --run-name general_sentence_candidate `
  --artifact-root experiments\general_sentence_candidate `
  --candidate-pools 10 25 50 75 100
```

1k candidate-recall run:

```powershell
python .\run_larp_experiment.py `
  --n-docs 1000 `
  --anchor-count 128 `
  --batch-size 16 `
  --max-tokens 160 `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --run-name large_minilm_candidate `
  --artifact-root experiments\large_minilm_candidate `
  --candidate-pools 25 50 100 250 500
```

10k candidate-recall run in a separate artifact folder:

```powershell
python .\run_larp_experiment.py `
  --n-docs 10000 `
  --anchor-count 256 `
  --batch-size 32 `
  --max-tokens 160 `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --run-name tenk_minilm_candidate `
  --artifact-root experiments\tenk_minilm_candidate `
  --skip-cross-model `
  --skip-anchor-count `
  --skip-perturbation `
  --sample-queries 1000 `
  --candidate-pools 50 100 250 500 1000
```

10k speed benchmark:

```powershell
python .\scripts\07_signature_index_benchmark.py
```

10k anchor/signature ablation:

```powershell
python .\scripts\08_anchor_ablation.py
```

Focused tight-pool ablation for the strongest strategies:

```powershell
python .\scripts\08_anchor_ablation.py `
  --out-dir experiments\tenk_minilm_candidate\anchor_ablation_tight `
  --strategies farthest_random kmeans_boundary density_sparse multi_scale random `
  --transforms row_zscore raw rank `
  --pools 25 50 75 100 250
```

End-to-end LARP index demo:

```powershell
python .\scripts\09_larp_index_demo.py
```

Multi-model robustness report from cached embeddings:

```powershell
python .\scripts\10_model_robustness.py `
  --embedding-dir experiments\robustness_3k_general_rerun\outputs\embeddings `
  --out-dir experiments\robustness_3k_general_rerun\model_robustness `
  --n-docs 3000 `
  --anchor-count 256 `
  --sample-queries 1000 `
  --top-k 10 `
  --pools 50 100 250 500 1000
```

Learned anchor-selection experiment:

```powershell
python .\scripts\11_learned_anchor_selection.py `
  --embedding-dir experiments\robustness_3k_general_rerun\outputs\embeddings `
  --out-dir experiments\learned_anchor_selection `
  --n-docs 3000 `
  --train-docs 2000 `
  --train-queries 800 `
  --eval-queries 800 `
  --candidate-anchor-count 512 `
  --anchor-count 256 `
  --steps 250 `
  --batch-size 128 `
  --neg-count 32 `
  --hard-pool 500 `
  --top-k 10 `
  --pools 10 25 50 100 250 500 1000
```

LARP-HNSW system benchmark:

```powershell
python .\scripts\12_larp_hnsw_system_benchmark.py `
  --embedding-path experiments\tenk_minilm_candidate\outputs\embeddings\sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy `
  --corpus-path experiments\tenk_minilm_candidate\data\processed\hf_google_code_x_glue_ct_code_to_text_n10000_seed7.jsonl `
  --out-dir experiments\larp_hnsw_system `
  --build-docs 8000 `
  --insert-batches 500 500 1000 `
  --query-count 300 `
  --anchor-count 256 `
  --pools 100 250 500 `
  --ef-search 128
```

Learned-anchor seed report:

```powershell
python .\scripts\13_learned_anchor_seed_report.py `
  --runs 7=experiments\learned_anchor_selection\learned_anchor_selection.csv `
         17=experiments\learned_anchor_selection_seed17\learned_anchor_selection.csv `
         29=experiments\learned_anchor_selection_seed29\learned_anchor_selection.csv `
  --out-dir experiments\learned_anchor_seed_report `
  --pool 250
```

BEIR SciFact labeled relevance benchmark:

```powershell
python .\scripts\14_beir_scifact_benchmark.py `
  --data-dir data\external\beir `
  --out-dir experiments\beir_scifact `
  --model-name sentence-transformers/all-MiniLM-L6-v2 `
  --batch-size 32 `
  --max-tokens 160 `
  --anchor-count 256 `
  --pools 100 250 500 `
  --top-k 10
```

Bilinear relation-only metric:

```powershell
python .\scripts\15_bilinear_relation_metric.py `
  --embedding-path experiments\tenk_minilm_candidate\outputs\embeddings\sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy `
  --out-dir experiments\bilinear_relation_metric `
  --n-docs 10000 `
  --train-docs 8000 `
  --train-queries 2000 `
  --eval-queries 1000 `
  --anchor-count 256 `
  --proj-dim 256 `
  --steps 1000 `
  --batch-size 256 `
  --neg-count 64 `
  --hard-pool 500 `
  --lr 0.001 `
  --identity-reg 0.1 `
  --ridge-reg 1.0 `
  --pools 10 25 50 100 250
```

Two-hour ridge-bilinear sweep for relation-only top-k:

```powershell
python .\scripts\16_ridge_bilinear_sweep.py `
  --embedding-path experiments\tenk_minilm_candidate\outputs\embeddings\sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy `
  --out-dir experiments\ridge_bilinear_sweep `
  --n-docs 10000 `
  --train-docs 8000 `
  --eval-queries 1000 `
  --anchor-counts 256 384 512 768 1024 `
  --ridge-regs 0.001 0.003 0.01 0.03 0.1 0.3 1.0 `
  --transforms row_zscore raw_l2 `
  --targets raw `
  --pools 10 25 50 100 250 `
  --time-budget-minutes 115
```

Small/100k/200k relation-index scale benchmark:

```powershell
python .\scripts\17_scale_relation_index.py `
  --embedding-path experiments\tenk_minilm_candidate\outputs\embeddings\sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy `
  --out-dir experiments\relation_index_scale `
  --sizes 1000 3000 10000 100000 200000 `
  --anchor-count 1024 `
  --anchor-candidate-docs 20000 `
  --ridge-reg 0.03 `
  --eval-queries 500 `
  --mechanics-queries 500 `
  --pools 10 25 `
  --batch-size 20000
```

## Detailed Experiment Log

The sections below preserve the progression of results. Some earlier results were superseded by later stricter baselines, especially the overlapping 100k chunk run and the raw-HNSW comparison.

On the 10k MiniLM run with 256 fixed anchors and 1,000 sampled queries, relative-signature candidate pools recovered the raw embedding top-10 at:

| Candidate Pool | Mean Recall of Raw Top-10 |
|---:|---:|
| 50 | 0.8497 |
| 100 | 0.9130 |
| 250 | 0.9643 |
| 500 | 0.9825 |
| 1000 | 0.9925 |

At 10k documents, exact vectorized search over relative signatures was faster than a Python tree prototype:

| Method | Query Time |
|---|---:|
| Vectorized exact relative search | ~0.084 ms/query |
| Random-projection forest prototype | ~4.92 ms/query |

So for 10k, brute vectorized relative search is already sufficient. At larger scale, use a compiled ANN index such as HNSW/FAISS/ScaNN rather than a Python object tree.

The best anchor/signature ablation so far is `farthest_random + row_zscore`:

| Candidate Pool | Mean Recall of Raw Top-10 |
|---:|---:|
| 25 | 0.8234 |
| 50 | 0.9065 |
| 100 | 0.9583 |
| 250 | 0.9850 |

The concrete `LARPIndex` demo builds on 9,900 docs, inserts 100 held-out docs, saves/loads the index, and searches 1,000 sampled queries:

| Metric | Value |
|---|---:|
| Build time | 0.5519 s |
| Batch insert time | 0.2356 ms/doc |
| Full search + raw rerank, pool 500 | 2.2815 ms/query |
| Candidate recall, pool 100 | 0.9554 |
| Candidate recall, pool 250 | 0.9850 |

The completed six-model 3k-document robustness rerun used `farthest_random + row_zscore` with 256 anchors and 1,000 sampled queries per model:

| Model | Pool 50 | Pool 100 | Pool 250 | Pool 500 |
|---|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | 0.8370 | 0.9201 | 0.9742 | 0.9923 |
| intfloat/e5-small-v2 | 0.8486 | 0.9349 | 0.9857 | 0.9974 |
| sentence-transformers/all-MiniLM-L12-v2 | 0.8729 | 0.9355 | 0.9824 | 0.9962 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.9015 | 0.9550 | 0.9889 | 0.9983 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 0.8962 | 0.9596 | 0.9935 | 0.9993 |
| sentence-transformers/paraphrase-albert-small-v2 | 0.6856 | 0.7908 | 0.9008 | 0.9625 |

This is evidence against a pure MiniLM fluke: MiniLM, E5, and BGE all perform well. It is not universal yet: ALBERT needs a much larger candidate pool to reach the same recall band.

Learned anchor selection improves the routing layer. The experiment trains a lightweight anchor-weight selector over a 512-anchor bank from the first 2,000 docs, keeps the top 256 anchors, and evaluates on 800 held-out queries from the last 1,000 docs:

| Strategy | Mean Pool 10 | Mean Pool 100 | Mean Pool 250 |
|---|---:|---:|---:|
| random_row_zscore | 0.4821 | 0.8809 | 0.9517 |
| farthest_row_zscore | 0.5080 | 0.9051 | 0.9659 |
| learned_top256 | 0.5511 | 0.9544 | 0.9902 |
| learned_weighted_bank | 0.5426 | 0.9592 | 0.9936 |

This is a real upgrade for candidate generation, especially at smaller pools. It still does not make relation-only top-10 a complete replacement: pool 10 mean recall is about 0.55, while pool 100-250 is where the method becomes very strong.

The HNSW system benchmark is the strongest reality check. On 10k MiniLM embeddings, built on 8k docs and incrementally inserted 2k docs:

| Method | Pool | Recall of Exact Raw Top-10 | ms/query |
|---|---:|---:|---:|
| raw_hnsw | 10 | 1.0000 | 1.195 |
| larp_hnsw_rerank | 100 | 0.9513 | 1.735 |
| larp_hnsw_rerank | 250 | 0.9850 | 2.466 |
| larp_hnsw_rerank | 500 | 0.9940 | 4.077 |
| larp_hnsw_relation_only | 10 | 0.6383 | 1.340 |
| multi_generation_larp | 500 | 0.9963 | 6.092 |

At this scale and with this implementation, raw HNSW is the better final search backend. LARP's defensible role is a trainable, compact routing layer, not a raw-HNSW replacement.

On BEIR SciFact, a real labeled retrieval benchmark, LARP-HNSW with reranking nearly matches raw exact/HNSW by pool 500:

| Method | Pool | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| raw_exact | 10 | 0.7817 | 0.5840 | 0.6292 |
| raw_hnsw | 10 | 0.7817 | 0.5840 | 0.6292 |
| larp_relation_only | 10 | 0.6314 | 0.4356 | 0.4774 |
| larp_hnsw_rerank | 100 | 0.7510 | 0.5750 | 0.6140 |
| larp_hnsw_rerank | 250 | 0.7667 | 0.5813 | 0.6228 |
| larp_hnsw_rerank | 500 | 0.7750 | 0.5829 | 0.6269 |

The first relation-only top-k breakthrough came from ridge-distilled bilinear scoring. This learns a projection from relation signatures back toward raw embedding geometry on the first 8k docs, then evaluates relation-only retrieval on 1k held-out later-doc queries:

| Method | Pool 10 | Pool 25 | Pool 50 | Pool 100 | Pool 250 |
|---|---:|---:|---:|---:|---:|
| cosine_relation | 0.6236 | 0.8176 | 0.9033 | 0.9534 | 0.9841 |
| contrastive_bilinear | 0.5931 | 0.7826 | 0.8683 | 0.9215 | 0.9673 |
| ridge_bilinear | 0.8314 | 0.9846 | 0.9975 | 0.9995 | 1.0000 |

So plain contrastive bilinear hurt, but ridge-distilled bilinear made relation-only top-10 much more plausible.

Scale benchmark summary:

| Docs | Mode | Relation-Only Pool-10 Recall | HNSW Build | HNSW ms/query | Vector MB |
|---:|---|---:|---:|---:|---:|
| 1,000 | real subset | 0.8670 | 0.06 s | 0.178 | 1.5 |
| 3,000 | real subset | 0.9070 | 0.17 s | 0.542 | 4.6 |
| 10,000 | real subset | 0.9224 | 1.11 s | 0.647 | 15.4 |
| 100,000 | synthetic mechanics | n/a | 20.51 s | 0.570 | 153.6 |
| 200,000 | synthetic mechanics | n/a | 42.15 s | 0.505 | 307.2 |

The 100k/200k rows are mechanics-only because they use noisy resampling of the 10k embeddings. A real quality test at those sizes requires embedding 100k/200k real chunks first.

A real 100k chunk run has now been completed from overlapping chunks of the 10k CodeXGLUE-derived snippet corpus:

| Docs | Relation-Only Pool-10 Recall | Relation-Only Pool-25 Recall | HNSW Build | HNSW ms/query | Vector MB |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.9804 | 1.0000 | 10.87 s | 0.115 | 153.6 |

This is a stronger relation-only result than the earlier synthetic mechanics test, but it should be read with the chunking caveat: the 100k rows are real chunks, not 100k independent source files. Reproduce with:

```powershell
scripts\run_real_100k_relation_test.cmd
```

The stricter result is the real distinct CodeXGLUE Python run. This uses Hugging Face Parquet rows directly: one dataset row per embedded item, deduped by code text hash, with no overlap chunking and no synthetic expansion.

| Docs | Relation-Only Pool-10 Recall | Relation-Only Pool-25 Recall | HNSW Build | HNSW ms/query | Vector MB |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.9638 | 1.0000 | 21.89 s | 0.184 | 153.6 |
| 200,000 | 0.9614 | 1.0000 | 51.83 s | 0.201 | 307.2 |

Reproduce the full distinct dataset build, embed, and scale run with:

```powershell
scripts\run_real_distinct_200k_test.cmd
```

If the corpora are already built and you only need to resume embedding/evaluation:

```powershell
scripts\run_real_distinct_200k_embed_scale.cmd
```

Against direct raw vector search, the relation backend is not superior as top-10 search:

| Docs | Method | Recall of Exact Raw Top-10 | All Top-10 Contained | Build | ms/query |
|---:|---|---:|---:|---:|---:|
| 100,000 | raw_hnsw | 0.9978 | 0.9840 | 29.44 s | 0.365 |
| 100,000 | ridge_relation_hnsw | 0.9550 | 0.5940 | 23.58 s + 22.30 s prep | 0.196 |
| 200,000 | raw_hnsw | 0.9924 | 0.9560 | 56.13 s | 0.224 |
| 200,000 | ridge_relation_hnsw | 0.9584 | 0.6260 | 57.10 s + 28.51 s prep | 0.226 |

At pool 25, ridge-relation HNSW nearly matches raw HNSW as a candidate generator, but raw HNSW is the better direct search backend. Reproduce this check with:

```powershell
.venv\Scripts\python.exe scripts\21_raw_vector_search_baseline.py
```

## Cleanup Note

The repo already had generated artifacts tracked before this cleanup. `.gitignore` prevents future generated data from being added, but it does not untrack files already in git.

To untrack generated files while keeping them on disk, run only if you are ready for that git index change:

```powershell
git rm --cached -r data outputs __pycache__ scripts\__pycache__
git add .gitignore README.md requirements.txt results.md reports scripts run_larp_experiment.py config.yaml
```

Review with:

```powershell
git status --short
```
