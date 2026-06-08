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

## Bottom Line

The hypothesis is partly alive, but not in the strongest form.

What looks promising:

- Relative signatures consistently improve broad pairwise geometry preservation for general sentence models.
- Small anchor counts can approximate same-model retrieval surprisingly well.
- Fixed-anchor signatures are stable under corpus additions.

What looks bad:

- Cross-model nearest-neighbor preservation is worse with relative signatures in every tested pair.
- Code-specific models are less favorable than general sentence models.
- The result does not reproduce the high MRR preservation reported in the original relative-representation work.

Practical implication:

Relative anchor signatures may be useful as a coarse, stable routing or clustering layer for an incremental semantic index. They do not yet look strong enough to replace exact embedding-space nearest-neighbor search for final retrieval.

## Artifacts

Reports:

- `larp_results.md`
- `larp_results_general_sentence_models.md`

Tables:

- `outputs/tables/cross_model_preservation.csv`
- `outputs/tables/anchor_count_curve.csv`
- `outputs/tables/corpus_perturbation.csv`
- `outputs/tables/general_sentence_models_cross_model_preservation.csv`
- `outputs/tables/general_sentence_models_anchor_count_curve.csv`
- `outputs/tables/general_sentence_models_corpus_perturbation.csv`

Runner:

- `run_larp_experiment.py`
