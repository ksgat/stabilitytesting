# Relative Embedding Preservation Results

## Setup

- Corpus: 120 CodeXGLUE code-to-text snippets
- Language mix: 20 each from Go, Java, JavaScript, PHP, Python, and Ruby
- Signature method: cosine similarity to 32 random in-corpus anchors
- Metrics:
  - MRR@10 preservation: nearest-neighbor agreement between two models
  - Spearman rho: rank correlation of pairwise distance structure
  - Drift: cosine drift of existing-document signatures under corpus expansion

The experiment was intentionally small enough to run locally on CPU and stay within a modest download budget. Treat it as a directional test, not a statistically final result.

## Headline

Anchor-relative signatures preserve broad geometry better than raw absolute spaces in many cases, especially for general sentence embedding models. They do not preserve exact nearest-neighbor retrieval better in these runs.

In both model suites, relative signatures lost MRR@10 against absolute embeddings for every tested model pair.

However, a later candidate-generation test is much more favorable to the index idea: relative signatures can retrieve a candidate pool that contains most or nearly all of the raw-embedding top-k before reranking.

## Code-Oriented Model Suite

Models:

- `microsoft/codebert-base`
- `Salesforce/codet5-small`
- `microsoft/unixcoder-base`

Cross-model preservation:

| Model A | Model B | Abs MRR | Rel MRR | Delta | Abs rho | Rel rho |
|---|---|---:|---:|---:|---:|---:|
| codebert-base | codet5-small | 0.8669 | 0.6171 | -0.2498 | 0.7436 | 0.7116 |
| codebert-base | unixcoder-base | 0.7513 | 0.5154 | -0.2359 | 0.3242 | 0.4112 |
| codet5-small | unixcoder-base | 0.7825 | 0.5745 | -0.2079 | 0.3725 | 0.5155 |

Interpretation:

The relative representation helped Spearman rank correlation for two of the three code-model pairs, but it substantially hurt nearest-neighbor preservation. This is not a clean replication of Moschella-style cross-model retrieval preservation on code embeddings.

Figures:

- `outputs/figures/cross_model_mrr.png`
- `outputs/figures/anchor_count_curve.png`
- `outputs/figures/corpus_perturbation.png`

## General Sentence Embedding Suite

Models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-MiniLM-L12-v2`
- `sentence-transformers/paraphrase-MiniLM-L3-v2`
- `sentence-transformers/paraphrase-albert-small-v2`
- `intfloat/e5-small-v2`
- `BAAI/bge-small-en-v1.5`

Cross-model preservation summary:

- Relative Spearman improved for all 15 model pairs.
- Relative MRR declined for all 15 model pairs.
- MRR losses were smallest for closely related MiniLM pairs.
- MRR losses were largest when ALBERT was involved.

Representative pairs:

| Model A | Model B | Abs MRR | Rel MRR | Delta | Abs rho | Rel rho |
|---|---|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | all-MiniLM-L12-v2 | 0.9402 | 0.8922 | -0.0480 | 0.5392 | 0.7641 |
| all-MiniLM-L6-v2 | paraphrase-MiniLM-L3-v2 | 0.8680 | 0.8353 | -0.0326 | 0.4670 | 0.6562 |
| all-MiniLM-L12-v2 | e5-small-v2 | 0.9018 | 0.8543 | -0.0475 | 0.3768 | 0.8103 |
| paraphrase-albert-small-v2 | e5-small-v2 | 0.7596 | 0.4979 | -0.2618 | 0.1695 | 0.3627 |
| e5-small-v2 | bge-small-en-v1.5 | 0.8997 | 0.8050 | -0.0946 | 0.2887 | 0.8080 |

Interpretation:

This suite is much more favorable to the relative-representation idea at the global-geometry level. The Spearman gains are often large. But the nearest-neighbor result still says the relative signature is smoothing or distorting local neighborhoods enough to hurt retrieval.

Figures:

- `outputs/figures/general_sentence_models_cross_model_mrr.png`
- `outputs/figures/general_sentence_models_anchor_count_curve.png`
- `outputs/figures/general_sentence_models_corpus_perturbation.png`

## Corpus Perturbation

With fixed anchors, signature drift is exactly zero in this setup. That is expected: snippets are embedded independently, and adding unrelated documents does not change existing embeddings or existing anchor vectors.

The more meaningful test is refitting anchors after corpus growth:

| Suite | Drift at +25% | Drift at +100% |
|---|---:|---:|
| Code-oriented models | 0.000505 | 0.000637 |
| General sentence models | 0.012611 | 0.012012 |

Interpretation:

Fixed-anchor indexing is stable by construction. Re-selecting anchors creates drift, but the observed drift is still small in these runs.

## Anchor Count

Code-oriented primary model, `codebert-base`:

| Anchors | MRR vs Raw | Spearman | Top-1 Overlap |
|---:|---:|---:|---:|
| 8 | 0.8034 | 0.9035 | 0.2500 |
| 16 | 0.9260 | 0.9182 | 0.2667 |
| 32 | 0.9454 | 0.9235 | 0.3167 |

General sentence primary model, `all-MiniLM-L6-v2`:

| Anchors | MRR vs Raw | Spearman | Top-1 Overlap |
|---:|---:|---:|---:|
| 8 | 0.8156 | 0.3405 | 0.3167 |
| 16 | 0.8609 | 0.3823 | 0.4583 |
| 32 | 0.9584 | 0.5702 | 0.5667 |

Interpretation:

The anchor-count curves are encouraging inside a single model: 32 anchors recovered high MRR against raw-space retrieval in both suites. The issue is not whether anchors can summarize one model's space. The issue is whether those summaries transfer local neighborhoods across models.

## Candidate Recall For Indexing

This is the most relevant experiment for a concrete incremental index.

Question:

Can a relative-signature index retrieve a candidate pool that contains the raw embedding model's true top-k neighbors, even if the relative-signature ordering itself is imperfect?

For a 1,000-document run with `sentence-transformers/all-MiniLM-L6-v2`, 128 fixed anchors, and raw top-10 as the target:

| Candidate Pool | Mean Recall of Raw Top-10 | All Top-10 Contained | Any Hit |
|---:|---:|---:|---:|
| 25 | 0.8061 | 0.3010 | 1.0000 |
| 50 | 0.8958 | 0.5040 | 1.0000 |
| 100 | 0.9543 | 0.7270 | 1.0000 |
| 250 | 0.9920 | 0.9360 | 1.0000 |
| 500 | 0.9995 | 0.9960 | 1.0000 |

For raw top-5:

| Candidate Pool | Mean Recall of Raw Top-5 | All Top-5 Contained | Any Hit |
|---:|---:|---:|---:|
| 25 | 0.8880 | 0.6500 | 0.9990 |
| 50 | 0.9418 | 0.7930 | 1.0000 |
| 100 | 0.9764 | 0.9050 | 1.0000 |
| 250 | 0.9970 | 0.9850 | 1.0000 |
| 500 | 0.9998 | 0.9990 | 1.0000 |

For raw top-1:

| Candidate Pool | Recall of Raw Top-1 |
|---:|---:|
| 25 | 0.9600 |
| 50 | 0.9720 |
| 100 | 0.9890 |
| 250 | 0.9980 |
| 500 | 1.0000 |

Interpretation:

This is strong evidence for using relative signatures as a first-stage candidate-generation index. The relative index does not need to return the final exact top-k. It needs to return a small candidate pool that almost always contains the true raw-embedding neighbors, after which raw cosine similarity can rerank the pool.

Artifacts:

- `reports/larp_results_large_minilm_candidate.md`
- `outputs/tables/large_minilm_candidate_candidate_recall.csv`
- `outputs/figures/large_minilm_candidate_candidate_recall.png`

For a larger 10,000-document run with `sentence-transformers/all-MiniLM-L6-v2`, 256 fixed anchors, and 1,000 sampled queries:

| Candidate Pool | Mean Recall of Raw Top-10 | All Top-10 Contained | Any Hit |
|---:|---:|---:|---:|
| 50 | 0.8497 | 0.4070 | 1.0000 |
| 100 | 0.9130 | 0.5730 | 1.0000 |
| 250 | 0.9643 | 0.7890 | 1.0000 |
| 500 | 0.9825 | 0.8810 | 1.0000 |
| 1000 | 0.9925 | 0.9430 | 1.0000 |

For raw top-5:

| Candidate Pool | Mean Recall of Raw Top-5 | All Top-5 Contained | Any Hit |
|---:|---:|---:|---:|
| 50 | 0.9130 | 0.7190 | 0.9990 |
| 100 | 0.9512 | 0.8280 | 1.0000 |
| 250 | 0.9800 | 0.9220 | 1.0000 |
| 500 | 0.9904 | 0.9620 | 1.0000 |
| 1000 | 0.9954 | 0.9780 | 1.0000 |

For raw top-1:

| Candidate Pool | Recall of Raw Top-1 |
|---:|---:|
| 50 | 0.9620 |
| 100 | 0.9780 |
| 250 | 0.9900 |
| 500 | 0.9930 |
| 1000 | 0.9970 |

10k artifacts:

- `experiments/tenk_minilm_candidate/larp_results_tenk_minilm_candidate.md`
- `experiments/tenk_minilm_candidate/outputs/tables/tenk_minilm_candidate_candidate_recall.csv`
- `experiments/tenk_minilm_candidate/outputs/figures/tenk_minilm_candidate_candidate_recall.png`

## Search Speed And Tree Traversal

A quick tree-style benchmark was run on the 10k relative signatures.

Methods:

- `exact_relative`: vectorized matrix multiply over all 10k relative signatures
- `rp_forest`: a simple B-tree-like random-projection forest over relative signatures, traversing two leaves per tree and reranking collected candidates
- `sklearn` brute/KD/Ball tree probes over normalized relative signatures

Result:

| Method | Query Time | Notes |
|---|---:|---|
| Vectorized exact relative search | ~0.084 ms/query | Batched NumPy matrix multiply; fastest observed path |
| RP forest prototype | ~4.92 ms/query | Scored ~2,714 candidates/query; Python traversal overhead dominates |
| sklearn brute euclidean | ~1.47 ms/query | C-backed brute path |
| sklearn KD-tree | ~9.07 ms/query | High-dimensional tree traversal loses |
| sklearn BallTree | ~6.64 ms/query | Also slower than brute |

The RP forest recovered reasonable candidate recall, but it was slower than exact vectorized search at 10k:

| Method | Pool | Mean Recall of Raw Top-10 | All Top-10 Contained |
|---|---:|---:|---:|
| exact_relative | 100 | 0.9173 | 0.5630 |
| exact_relative | 250 | 0.9660 | 0.7890 |
| exact_relative | 500 | 0.9835 | 0.8890 |
| rp_forest | 100 | 0.8893 | 0.4710 |
| rp_forest | 250 | 0.9257 | 0.5870 |
| rp_forest | 500 | 0.9365 | 0.6240 |

Interpretation:

For 10k documents, a tree is not needed. Exact vectorized relative search is already faster and more accurate. The B-tree-like structure is conceptually possible, but the useful production version should likely be a compiled ANN structure such as HNSW, IVF/PQ, or a custom vectorized bucket/projection index rather than a Python tree.

Tree benchmark artifacts:

- `scripts/07_signature_index_benchmark.py`
- `experiments/tenk_minilm_candidate/tree_index/signature_index_benchmark.md`
- `experiments/tenk_minilm_candidate/tree_index/signature_index_benchmark.csv`
- `experiments/tenk_minilm_candidate/tree_index/signature_index_benchmark.png`

## Anchor Strategy Ablation

A broad 10k ablation tested anchor strategy and signature transform combinations using the same `all-MiniLM-L6-v2` embeddings, 256 anchors, 1,000 sampled queries, and raw top-10 as the target.

Tested anchor families:

- random
- language-stratified random
- farthest-point from random seed
- farthest-point from corpus centroid
- PCA extremes
- kmeans medoids
- kmeans boundary points
- language-balanced kmeans medoids
- dense, sparse, and mixed density anchors
- multi-scale anchors
- hard-negative anchors

Tested signature transforms:

- raw cosine-to-anchor
- column-centered
- global zscore
- row zscore
- rank
- top-32 and top-64 sparse
- binary top-32
- softmax temperatures `0.05` and `0.1`
- centered sign bits

Best broad-grid results at pool 250:

| Anchor Strategy | Transform | Mean Recall of Raw Top-10 | All Top-10 Contained |
|---|---|---:|---:|
| farthest_random | row_zscore | 0.9879 | 0.9030 |
| farthest_random | raw | 0.9839 | 0.8810 |
| kmeans_boundary | row_zscore | 0.9817 | 0.8730 |
| farthest_centroid | row_zscore | 0.9804 | 0.8610 |
| multi_scale | row_zscore | 0.9797 | 0.8560 |
| density_sparse | row_zscore | 0.9790 | 0.8480 |

Focused tight-pool results:

| Pool | Best Strategy | Best Transform | Mean Recall of Raw Top-10 | All Top-10 Contained |
|---:|---|---|---:|---:|
| 25 | farthest_random | row_zscore | 0.8234 | 0.3170 |
| 50 | farthest_random | row_zscore | 0.9065 | 0.5320 |
| 100 | farthest_random | row_zscore | 0.9583 | 0.7260 |
| 250 | farthest_random | row_zscore | 0.9850 | 0.8890 |

Interpretation:

The strongest practical fix so far is not kmeans-medoid. It is diversity-heavy anchor coverage plus row-wise signature normalization. Dense anchors performed poorly, while sparse/outlier/boundary anchors performed well. This suggests the relative index benefits from anchors that span tails and decision boundaries rather than anchors that summarize the densest regions.

Anchor ablation artifacts:

- `scripts/08_anchor_ablation.py`
- `experiments/tenk_minilm_candidate/anchor_ablation/anchor_ablation.md`
- `experiments/tenk_minilm_candidate/anchor_ablation/anchor_ablation.csv`
- `experiments/tenk_minilm_candidate/anchor_ablation_tight/anchor_ablation.md`
- `experiments/tenk_minilm_candidate/anchor_ablation_tight/anchor_ablation.csv`

## Concrete LARP Index Demo

A minimal `LARPIndex` implementation now exists. It uses:

- farthest-point anchors
- row-zscored relative signatures
- exact vectorized relative candidate search
- raw embedding cosine rerank
- batch insertion
- save/load

Demo setup:

- 9,900 base documents
- 100 held-out documents inserted after build
- 10,000 total documents after insert
- 256 anchors
- 1,000 sampled queries

Measured result:

| Metric | Value |
|---|---:|
| Build time | 0.5519 s |
| Batch insert time | 0.2356 ms/doc |
| Save time | 0.0557 s |
| Load time | 0.1728 s |
| Full search + raw rerank at pool 500 | 2.2815 ms/query |

Candidate recall after insert:

| Pool | Mean Recall of Raw Top-10 | All Top-10 Contained |
|---:|---:|---:|
| 50 | 0.9053 | 0.5260 |
| 100 | 0.9554 | 0.7200 |
| 250 | 0.9850 | 0.8830 |
| 500 | 0.9947 | 0.9520 |

Index demo artifacts:

- `larp_index.py`
- `scripts/09_larp_index_demo.py`
- `experiments/tenk_minilm_candidate/larp_index_demo/larp_index_demo.md`
- `experiments/tenk_minilm_candidate/larp_index_demo/larp_index_demo_metrics.csv`

## Multi-Model Robustness Check

I reran the current best candidate-generation test over a fresh six-model, 3k-document embedding sweep. The test uses each model independently:

1. Normalize raw embeddings.
2. Select 256 fixed farthest-point anchors.
3. Convert each document to anchor-relative cosine signatures.
4. Apply per-document row z-score normalization.
5. Retrieve candidates by relative-signature cosine similarity.
6. Measure whether each model's raw embedding top-10 neighbors appear in the candidate pool.

This does not test direct replacement of raw embedding search. It tests whether the relative signature is a stable candidate generator across models.

Best setting: `farthest_random + row_zscore`, 1,000 sampled queries per model.

| Model | Dim | Pool 50 | Pool 100 | Pool 250 | Pool 500 | Pool 1000 |
|---|---:|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | 384 | 0.8370 | 0.9201 | 0.9742 | 0.9923 | 0.9991 |
| intfloat/e5-small-v2 | 384 | 0.8486 | 0.9349 | 0.9857 | 0.9974 | 0.9996 |
| sentence-transformers/all-MiniLM-L12-v2 | 384 | 0.8729 | 0.9355 | 0.9824 | 0.9962 | 0.9998 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.9015 | 0.9550 | 0.9889 | 0.9983 | 0.9999 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 384 | 0.8962 | 0.9596 | 0.9935 | 0.9993 | 1.0000 |
| sentence-transformers/paraphrase-albert-small-v2 | 768 | 0.6856 | 0.7908 | 0.9008 | 0.9625 | 0.9921 |

Strategy mean recall at pool 250:

| Strategy | Mean | Min | Max |
|---|---:|---:|---:|
| farthest_row_zscore | 0.9709 | 0.9008 | 0.9935 |
| farthest_raw | 0.9656 | 0.8753 | 0.9887 |
| random_row_zscore | 0.9599 | 0.8708 | 0.9851 |
| random_raw | 0.9478 | 0.8414 | 0.9810 |

Finding:

The result is probably not a single MiniLM fluke: MiniLM variants, E5, and BGE all land near perfect recall by pool 250-500. It is not universal yet: ALBERT is materially weaker and needs pool 500-1000 to reach the same quality band. That suggests model-specific calibration is still required.

Artifacts:

- `scripts/10_model_robustness.py`
- `experiments/robustness_3k_general_rerun/larp_results_robustness_3k_general_rerun.md`
- `experiments/robustness_3k_general_rerun/model_robustness/model_robustness.md`
- `experiments/robustness_3k_general_rerun/model_robustness/model_robustness.csv`
- `experiments/robustness_3k_general_rerun/model_robustness/model_robustness_curves.png`

## Learned Anchor Selection

I tested whether anchor selection can be trained as a separate lightweight model. This is the direct follow-up to the question: can an optimized relation coordinate system fix the gap between relative signatures and top-k search?

Setup:

1. Use the fresh six-model, 3k-document embedding sweep.
2. Treat the first 2,000 documents as the base/old corpus.
3. Build a 512-anchor candidate bank from base documents using farthest-point selection.
4. Train one positive weight per candidate anchor with a contrastive hard-negative loss using only base-document positives and negatives.
5. Keep the top 256 weighted anchors as `learned_top256`.
6. Evaluate on 800 held-out queries from the final 1,000 documents.

This is a stronger incremental test than the earlier robustness table because the learned selector only sees base-document anchor candidates and is evaluated on later/new-document queries.

Pool-250 held-out recall:

| Model | Farthest | Learned Top-256 | Delta | Weighted Bank |
|---|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | 0.9714 | 0.9891 | +0.0178 | 0.9915 |
| intfloat/e5-small-v2 | 0.9812 | 0.9926 | +0.0114 | 0.9920 |
| sentence-transformers/all-MiniLM-L12-v2 | 0.9725 | 0.9945 | +0.0220 | 0.9943 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.9878 | 0.9959 | +0.0081 | 0.9954 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 0.9924 | 0.9991 | +0.0067 | 0.9980 |
| sentence-transformers/paraphrase-albert-small-v2 | 0.8903 | 0.9699 | +0.0796 | 0.9906 |

Strategy means across the six models:

| Strategy | Pool 10 | Pool 25 | Pool 50 | Pool 100 | Pool 250 |
|---|---:|---:|---:|---:|---:|
| random_row_zscore | 0.4821 | 0.6803 | 0.7960 | 0.8809 | 0.9517 |
| farthest_row_zscore | 0.5080 | 0.7103 | 0.8260 | 0.9051 | 0.9659 |
| learned_top256 | 0.5511 | 0.7753 | 0.8880 | 0.9544 | 0.9902 |
| learned_weighted_bank | 0.5426 | 0.7692 | 0.8906 | 0.9592 | 0.9936 |

Finding:

Training anchor selection materially improves LARP. It fixes much of the ALBERT weakness and improves every tested model at pool 250, even under the stricter base-only training split. The best practical interpretation is still candidate generation: learned anchors make pool 100-250 very strong. They do not make relation-only top-10 a full replacement for reranking; pool 10 mean recall is only 0.5511.

Artifacts:

- `scripts/11_learned_anchor_selection.py`
- `experiments/learned_anchor_selection/learned_anchor_selection.md`
- `experiments/learned_anchor_selection/learned_anchor_selection.csv`
- `experiments/learned_anchor_selection/learned_anchor_selection_pool250.png`

Seed generalization:

I reran learned anchor selection for seeds 7, 17, and 29 and aggregated pool-250 recall. The learned selector stayed ahead of farthest anchors on every model.

| Strategy | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| random_row_zscore | 0.9541 | 0.0428 | 0.8574 | 0.9889 |
| farthest_row_zscore | 0.9673 | 0.0353 | 0.8857 | 0.9940 |
| learned_top256 | 0.9901 | 0.0091 | 0.9698 | 0.9991 |
| learned_weighted_bank | 0.9935 | 0.0035 | 0.9845 | 0.9980 |

Artifacts:

- `scripts/13_learned_anchor_seed_report.py`
- `experiments/learned_anchor_seed_report/learned_anchor_seed_report.md`

## LARP-HNSW System Benchmark

I built the production-shaped benchmark that compares raw-vector HNSW to LARP-HNSW under incremental inserts.

Setup:

1. Use 10k MiniLM embeddings from the CodeXGLUE corpus.
2. Build the index on the first 8,000 documents.
3. Insert 500, 500, then 1,000 new documents.
4. Evaluate 300 sampled corpus queries at each stage.
5. Use exact raw cosine top-10 as the retrieval target.
6. Compare raw HNSW, LARP-HNSW with raw rerank, relation-only LARP, and a two-generation LARP search facade.

Final 10k stage:

| Method | Pool | Recall | All Top-10 | ms/query | Label Precision |
|---|---:|---:|---:|---:|---:|
| raw_hnsw | 10 | 1.0000 | 1.0000 | 1.195 | 0.8090 |
| larp_hnsw_rerank | 100 | 0.9513 | 0.7267 | 1.735 | 0.8183 |
| larp_hnsw_rerank | 250 | 0.9850 | 0.8933 | 2.466 | 0.8133 |
| larp_hnsw_rerank | 500 | 0.9940 | 0.9500 | 4.077 | 0.8110 |
| larp_hnsw_relation_only | 10 | 0.6383 | 0.0067 | 1.340 | 0.8243 |
| multi_generation_larp | 500 | 0.9963 | 0.9667 | 6.092 | 0.7963 |

Drift diagnostics:

| Stage | Top-Anchor Sim | Entropy | Anchor Gini |
|---|---:|---:|---:|
| build | 0.7682 | 5.5439 | 0.5189 |
| insert batch 1 | 0.7553 | 5.5439 | 0.6383 |
| insert batch 2 | 0.7570 | 5.5439 | 0.6211 |
| insert batch 3 | 0.7588 | 5.5439 | 0.5714 |

Compression estimate at 10k:

| Representation | MB |
|---|---:|
| raw float32 | 15.360 |
| signature float32 | 10.240 |
| signature float16 | 5.120 |
| signature int8 | 2.560 |

Finding:

Raw HNSW is the stronger final search backend at 10k in this implementation. LARP-HNSW reaches high recall as a routing layer, but it is slower once the Python raw-rerank pool is included. Relation-only top-k is still not close enough to replace reranking. The practical value is a trainable relative routing layer, generation protocol, and compact signature representation, not a direct raw-HNSW killer yet.

Artifacts:

- `larp_hnsw_index.py`
- `scripts/12_larp_hnsw_system_benchmark.py`
- `experiments/larp_hnsw_system/larp_hnsw_system_benchmark.md`
- `experiments/larp_hnsw_system/larp_hnsw_system_benchmark.csv`
- `experiments/larp_hnsw_system/larp_hnsw_drift.csv`

## BEIR SciFact Relevance Benchmark

I added a real labeled retrieval benchmark using BEIR SciFact. This removes the earlier weakness where all quality numbers were either raw-neighbor preservation or metadata-label proxies.

Setup:

1. Download BEIR SciFact.
2. Embed 5,183 corpus documents and 300 test queries with `sentence-transformers/all-MiniLM-L6-v2`.
3. Build raw HNSW and LARP-HNSW with 256 anchors.
4. Evaluate against qrels using Recall@10, MRR@10, and NDCG@10.

Results:

| Method | Pool | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| raw_exact | 10 | 0.7817 | 0.5840 | 0.6292 |
| raw_hnsw | 10 | 0.7817 | 0.5840 | 0.6292 |
| larp_relation_only | 10 | 0.6314 | 0.4356 | 0.4774 |
| larp_hnsw_rerank | 100 | 0.7510 | 0.5750 | 0.6140 |
| larp_hnsw_rerank | 250 | 0.7667 | 0.5813 | 0.6228 |
| larp_hnsw_rerank | 500 | 0.7750 | 0.5829 | 0.6269 |

Finding:

On a labeled benchmark, LARP-HNSW with raw reranking nearly matches raw exact/HNSW by pool 500. Relation-only retrieval remains much worse. This supports the routing-layer claim and rejects the direct replacement claim.

Artifacts:

- `scripts/14_beir_scifact_benchmark.py`
- `experiments/beir_scifact/beir_scifact_benchmark.md`
- `experiments/beir_scifact/beir_scifact_benchmark.csv`

## Bilinear Relation-Only Metric

I tested the next hypothesis: the relation signatures contain the neighbor information, but plain cosine over signatures fails to rank the final top-k because it cannot model correlations between anchors.

Setup:

1. Use the 10k MiniLM embedding run.
2. Select 256 farthest anchors from the first 8,000 base documents.
3. Build row-zscore relation signatures for all 10k documents.
4. Train on 2,000 base-document queries.
5. Evaluate on 1,000 held-out queries from later documents.
6. No raw rerank at evaluation time.

Methods:

- `cosine_relation`: plain cosine over relation signatures.
- `contrastive_bilinear`: train `project(sig) = normalize(sig @ L)` with raw-neighbor positives and hard relation negatives.
- `ridge_bilinear`: solve a ridge regression from relation signatures to raw embeddings, then rank by cosine in the projected relation-only space. Scoring is still bilinear: `score(q,d) = sig(q)^T A A^T sig(d)`.

Results:

| Method | Pool 10 | Pool 25 | Pool 50 | Pool 100 | Pool 250 |
|---|---:|---:|---:|---:|---:|
| cosine_relation | 0.6236 | 0.8176 | 0.9033 | 0.9534 | 0.9841 |
| contrastive_bilinear | 0.5931 | 0.7826 | 0.8683 | 0.9215 | 0.9673 |
| ridge_bilinear | 0.8314 | 0.9846 | 0.9975 | 0.9995 | 1.0000 |

Finding:

This is the strongest evidence so far for relation-signature top-k. The naive contrastive bilinear metric overfits or distorts the held-out relation geometry, but ridge-distilled bilinear recovers much more of the raw top-k ordering without raw reranking. Relation-only top-10 is still not exact, but `0.8314` is a different regime than the earlier `0.55-0.64`.

Artifacts:

- `scripts/15_bilinear_relation_metric.py`
- `experiments/bilinear_relation_metric/bilinear_relation_metric.md`
- `experiments/bilinear_relation_metric/bilinear_relation_metric.csv`
- `experiments/bilinear_relation_metric/ridge_bilinear_projection.npy`

## Scale Test

I added a scale benchmark to separate actual smaller-set quality from 100k/200k mechanics. The 1k, 3k, and 10k rows use real subsets from the existing MiniLM embedding run. The 100k and 200k rows use noisy resampling of the existing 10k embeddings, so they are valid for build/search/memory mechanics but not quality claims.

Settings:

- 1,024 anchors
- ridge `0.03`
- relation-only ridge-bilinear projected vectors
- HNSW over projected relation vectors
- 500 eval/mechanics queries

Results:

| Docs | Mode | Pool-10 Recall | Pool-25 Recall | HNSW Build | HNSW ms/query | Vector MB |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | real subset | 0.8670 | 0.9985 | 0.06 s | 0.178 | 1.5 |
| 3,000 | real subset | 0.9070 | 0.9992 | 0.17 s | 0.542 | 4.6 |
| 10,000 | real subset | 0.9224 | 0.9992 | 1.11 s | 0.647 | 15.4 |
| 100,000 | synthetic mechanics | n/a | n/a | 20.51 s | 0.570 | 153.6 |
| 200,000 | synthetic mechanics | n/a | n/a | 42.15 s | 0.505 | 307.2 |

Finding:

The relation index mechanics are not scary at 100k-200k chunks. The expensive part for a real repository is embedding and chunk extraction, not relation projection or HNSW search. For quality, the available real subset results improve with size, reaching 0.9224 relation-only top-10 recall at 10k and nearly perfect pool-25 containment.

Artifacts:

- `scripts/17_scale_relation_index.py`
- `experiments/relation_index_scale/relation_index_scale.md`
- `experiments/relation_index_scale/relation_index_scale.csv`

## Real 100k Chunk Scale Test

I then built a real 100k chunk corpus from the existing CodeXGLUE-derived 10k snippet corpus using overlapping code chunks. This is not 100k unique repositories or files; it is 100k real text/code chunks derived from real snippets, which is the closer shape for a file-save/commit-time chunk index.

Settings:

- Source: `experiments/tenk_minilm_candidate/data/processed/hf_google_code_x_glue_ct_code_to_text_n10000_seed7.jsonl`
- Chunking: 220 chars, 30-char stride, 40-char minimum
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings: 100,000 x 384 float32, 153.6 MB
- Anchors: 1,024 farthest anchors from the training split
- Metric: row-zscored anchor relations projected with ridge-bilinear distillation, ridge `0.03`
- Evaluation: 500 held-out later-chunk queries per size, exact raw embedding top-10 as truth

Results:

| Docs | Method | Pool-10 Recall | Pool-25 Recall | Pool-10 All Top-10 | Pool-25 All Top-10 | HNSW Build | HNSW ms/query |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1,000 | ridge_bilinear | 0.9165 | 0.9875 | 0.4750 | 0.9200 | 0.05 s | 0.019 |
| 3,000 | ridge_bilinear | 0.9472 | 0.9996 | 0.5720 | 0.9960 | 0.12 s | 0.030 |
| 10,000 | ridge_bilinear | 0.9658 | 0.9998 | 0.7080 | 0.9980 | 0.62 s | 0.045 |
| 100,000 | ridge_bilinear | 0.9804 | 1.0000 | 0.8100 | 1.0000 | 10.87 s | 0.115 |

Finding:

The ridge-bilinear relation metric did not collapse at 100k. On this chunked corpus, relation-only top-k improved with scale: pool-10 recall reached `0.9804`, and pool-25 contained the full exact raw top-10 for all sampled 100k queries. That is the first result here that looks like a plausible relation-signature top-k index rather than just a broad candidate router.

Caveat:

The corpus has overlapping chunks from 10k source snippets, so it is easier than 100k fully independent documents. The next validation should use a larger independent corpus or a large repository/document dump with natural chunk boundaries.

Artifacts:

- `scripts/18_real_chunk_corpus_embed.py`
- `scripts/run_real_100k_relation_test.cmd`
- `scripts/run_real_100k_relation_test.ps1`
- `experiments/real_100k_chunks/relation_index_scale/relation_index_scale.md`
- `experiments/real_100k_chunks/relation_index_scale/relation_index_scale.csv`

## Real Distinct 100k/200k CodeXGLUE Test

To remove the overlap-chunk saturation issue, I pulled the Hugging Face `google/code_x_glue_ct_code_to_text` Python train Parquet shards and built corpora from distinct source rows. There is no noisy resampling and no overlapping chunk expansion in this run: one dataset row becomes one embedded item, deduped by SHA-256 of the code text.

Dataset:

- Source files: `python/train-00000-of-00002.parquet`, `python/train-00001-of-00002.parquet`
- Distinct processed corpora: 100,000 rows and 200,000 rows
- Scanned rows: 200,000
- Distinct rows accepted: 200,000
- 100k JSONL: 164.6 MB
- 200k JSONL: 331.4 MB
- 200k embedding matrix: 200,000 x 384 float32, 307.2 MB

Settings:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Max tokens: 128
- Anchors: 1,024 farthest anchors from the training split
- Metric: row-zscored anchor relations projected with ridge-bilinear distillation, ridge `0.03`
- Evaluation: 500 held-out later-row queries per size, exact raw embedding top-10 as truth

Results:

| Docs | Method | Pool-10 Recall | Pool-25 Recall | Pool-10 All Top-10 | Pool-25 All Top-10 | HNSW Build | HNSW ms/query |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100,000 | cosine_relation | 0.6038 | 0.7898 | 0.0120 | 0.2300 | n/a | n/a |
| 100,000 | ridge_bilinear | 0.9638 | 1.0000 | 0.6760 | 1.0000 | 21.89 s | 0.184 |
| 200,000 | cosine_relation | 0.5636 | 0.7630 | 0.0060 | 0.1720 | n/a | n/a |
| 200,000 | ridge_bilinear | 0.9614 | 1.0000 | 0.6440 | 1.0000 | 51.83 s | 0.201 |

Finding:

The saturated overlap result was too flattering, but the core signal survived. On 200k distinct real code examples, plain cosine over relation signatures is not enough. Ridge-bilinear relation projection is the difference-maker: relation-only pool-10 recall stays around `0.96` at both 100k and 200k, and pool-25 contains the full exact raw top-10 for every sampled query.

## Raw Vector Search Baseline

I then checked the same 100k/200k distinct CodeXGLUE embeddings against direct raw vector search. The truth set is exact cosine top-10 over raw MiniLM embeddings. This is the baseline the relation method has to beat if it is meant to be a primary search backend.

Settings:

- Same 100k and 200k distinct CodeXGLUE Python rows
- Same `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Raw HNSW: cosine, `M=32`, `ef_construction=200`, `ef_search=128`
- Relation HNSW: same HNSW params over ridge-bilinear projected relation vectors
- 500 held-out queries per size

Direct top-10 comparison:

| Docs | Method | Recall of Exact Raw Top-10 | All Top-10 Contained | Build | Prep | ms/query | Vector MB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100,000 | raw_exact | 1.0000 | 1.0000 | 0.00 s | 0.00 s | 1.469 | 153.6 |
| 100,000 | raw_hnsw | 0.9978 | 0.9840 | 29.44 s | 0.00 s | 0.365 | 153.6 |
| 100,000 | ridge_relation_hnsw | 0.9550 | 0.5940 | 23.58 s | 22.30 s | 0.196 | 153.6 |
| 200,000 | raw_exact | 1.0000 | 1.0000 | 0.00 s | 0.00 s | 2.858 | 307.2 |
| 200,000 | raw_hnsw | 0.9924 | 0.9560 | 56.13 s | 0.00 s | 0.224 | 307.2 |
| 200,000 | ridge_relation_hnsw | 0.9584 | 0.6260 | 57.10 s | 28.51 s | 0.226 | 307.2 |

Pool-25 candidate comparison:

| Docs | Method | Recall of Exact Raw Top-10 | All Top-10 Contained |
|---:|---|---:|---:|
| 100,000 | raw_hnsw | 0.9978 | 0.9840 |
| 100,000 | ridge_relation_hnsw | 0.9970 | 0.9740 |
| 200,000 | raw_hnsw | 0.9924 | 0.9560 |
| 200,000 | ridge_relation_hnsw | 0.9940 | 0.9540 |

Finding:

Raw HNSW wins as the primary top-10 search backend. Ridge-bilinear relation HNSW is not superior direct vector search: it has lower top-10 recall, similar memory, and extra projection/anchor prep cost. Its best remaining role is small-pool routing: at pool 25 it nearly matches raw HNSW containment of exact raw top-10 neighbors.

## relation_router_rerank_test

I changed the objective from direct nearest-neighbor replacement to product-style candidate routing. Each system produces candidates, then the same final cross-encoder reranker reranks the candidates. The question is whether relation routing wins on any operational axis: lower latency, smaller candidate pool, better reranked quality at a fixed budget, better update cost, or useful failure profile.

Setup:

- 100,000 distinct CodeXGLUE Python rows
- 100 held-out queries
- Relevance: exact raw-embedding top-10 neighbors
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Systems: low-ef raw HNSW, ridge-relation HNSW pool 25/50, BM25+dense hybrid, centroid/IVF routing

Results:

| System | Candidate Source | Pool | Recall@10 | MRR@10 | NDCG@10 | Candidate Containment | ms/query | Build | Update ms/doc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_hnsw_low_ef | raw vectors | 50 | 0.4200 | 0.8061 | 0.4800 | 0.9300 | 1315.119 | 30.60 s | 0.3141 |
| ridge_relation_pool25 | relation HNSW | 25 | 0.5500 | 0.8778 | 0.6008 | 0.9600 | 652.415 | 23.54 s | 0.3370 |
| ridge_relation_pool50 | relation HNSW | 50 | 0.4220 | 0.8048 | 0.4816 | 0.9600 | 1302.557 | 23.54 s | 0.3370 |
| bm25_dense_hybrid | BM25 + raw dense | 50 | 0.4630 | 0.8262 | 0.5202 | 1.0000 | 1330.860 | 8.01 s | n/a |
| cluster_ivf_baseline | centroid routing | 50 | 0.3330 | 0.7505 | 0.3905 | 0.2500 | 1427.986 | 41.03 s | 0.0095 |

Finding:

This is the narrowest positive result. `ridge_relation_pool25` beats every other tested router after cross-encoder reranking: better Recall@10, MRR@10, NDCG@10, strong candidate containment, and lower latency because it only reranks 25 candidates. The BM25+dense hybrid has perfect containment, but the MS MARCO cross-encoder does not turn that larger pool into better final quality here.

Limitation:

The cross-encoder is not code-specialized, so absolute final quality is low. The router comparison is still useful because every system uses the same reranker.

Artifacts:

- `scripts/22_relation_router_rerank_test.py`
- `scripts/run_relation_router_rerank_cross_encoder.cmd`
- `experiments/relation_router_rerank_cross_encoder/relation_router_rerank_test.md`
- `experiments/relation_router_rerank_cross_encoder/relation_router_rerank_summary.csv`
- `experiments/relation_router_rerank_cross_encoder/relation_router_rerank_failures.csv`

Artifacts:

- `scripts/19_build_distinct_hf_code_corpus.py`
- `scripts/20_embed_jsonl_corpus.py`
- `scripts/21_raw_vector_search_baseline.py`
- `scripts/run_real_distinct_200k_test.cmd`
- `scripts/run_real_distinct_200k_embed_scale.cmd`
- `experiments/real_distinct_hf_code/data/hf_code_x_glue_python_distinct_manifest.json`
- `experiments/real_distinct_hf_code/relation_index_scale/relation_index_scale.md`
- `experiments/real_distinct_hf_code/relation_index_scale/relation_index_scale.csv`
- `experiments/real_distinct_hf_code/raw_vector_baseline/raw_vector_search_baseline.md`
- `experiments/real_distinct_hf_code/raw_vector_baseline/raw_vector_search_baseline.csv`

## Ten-Step Completion Map

| Step | Status | Evidence |
|---:|---|---|
| 1. Baseline against real ANN | Done | Raw HNSW benchmark in `experiments/larp_hnsw_system/larp_hnsw_system_benchmark.md`. |
| 2. Real ANN backend for signatures | Done | `larp_hnsw_index.py` uses `hnswlib` for relative signatures and raw embeddings. |
| 3. Dynamic insert test | Done | 8k build plus 2k inserts in `scripts/12_larp_hnsw_system_benchmark.py`. |
| 4. Anchor drift detection | Done | `LARPHNSWIndex.drift_stats()` and `larp_hnsw_drift.csv`. |
| 5. Generation protocol | Done | `MultiGenerationLARP` and multi-generation benchmark row. |
| 6. Relation-only failure addressed | Done | Relation-only recall reported separately: 0.6383 at pool 10 in the final HNSW benchmark. |
| 7. Relevance benchmark | Done | BEIR SciFact benchmark in `experiments/beir_scifact/beir_scifact_benchmark.md`. |
| 8. Compression story | Done | Memory estimates for raw and signature formats. |
| 9. Learned anchor generalization | Done | Three-seed learned-anchor report in `experiments/learned_anchor_seed_report`. |
| 10. Product-shaped API | Done | `LARPHNSWIndex.fit_embeddings`, `insert_embeddings`, `search_embedding`, `search_text`, `save`, `load`; save/load verified locally. |

## Bottom Line

The hypothesis is partly alive, but not in the strongest form.

What looks promising:

- Relative signatures consistently improve broad pairwise geometry preservation for general sentence models.
- Small anchor counts can approximate same-model retrieval surprisingly well.
- Fixed-anchor signatures are stable under corpus additions.
- Relative-signature candidate pools recover raw-embedding top-k well enough to support a two-stage index design.

What looks bad:

- Cross-model nearest-neighbor preservation is worse with relative signatures in every tested pair.
- Code-specific models are less favorable than general sentence models.
- The result does not reproduce the high MRR preservation reported in the original relative-representation work.

Practical implication:

Relative anchor signatures look useful as a coarse, stable routing layer for an incremental semantic index. They do not look strong enough to replace exact embedding-space nearest-neighbor search for final retrieval, but they may be strong enough to make final retrieval cheap by narrowing the candidate set first.

Concrete search design:

1. Embed fixed anchors once.
2. For each document, store raw embedding and relative signature to the anchor set.
3. Build an ANN/HNSW index over relative signatures.
4. For each query, compute its relative signature and retrieve a candidate pool.
5. Rerank that pool with raw embedding cosine similarity.
6. Insert new documents by embedding only the new document and computing its relations to fixed anchors.

## Artifacts

Reports:

- `reports/larp_results.md`
- `reports/larp_results_code_models_candidate.md`
- `reports/larp_results_general_sentence_candidate.md`
- `reports/larp_results_general_sentence_models.md`
- `reports/larp_results_large_minilm_candidate.md`

Tables:

- `outputs/tables/cross_model_preservation.csv`
- `outputs/tables/anchor_count_curve.csv`
- `outputs/tables/corpus_perturbation.csv`
- `outputs/tables/general_sentence_models_cross_model_preservation.csv`
- `outputs/tables/general_sentence_models_anchor_count_curve.csv`
- `outputs/tables/general_sentence_models_corpus_perturbation.csv`
- `outputs/tables/code_models_candidate_candidate_recall.csv`
- `outputs/tables/general_sentence_candidate_candidate_recall.csv`
- `outputs/tables/large_minilm_candidate_candidate_recall.csv`

Runner:

- `run_larp_experiment.py`
