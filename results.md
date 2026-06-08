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
