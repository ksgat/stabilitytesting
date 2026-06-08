# Cross-Model Results

Run time: 2026-06-08 10:09:05 -04:00

Command run:

```bash
python .\run_larp_experiment.py --model-suite general-sentence --n-docs 120 --max-tokens 160 --batch-size 8 --anchor-count 32 --run-name general_sentence_models
```

## Run setup

- Corpus: 120 CodeXGLUE code snippets from `data/processed/hf_google_code_x_glue_ct_code_to_text_n120_seed7.jsonl`
- Language mix: 20 each from Java, PHP, Go, Ruby, JavaScript, and Python
- Models completed: 6 sentence embedding models
- Anchor count: 32
- Max tokens per snippet: 160
- Output tables:
  - `outputs/tables/general_sentence_models_cross_model_preservation.csv`
  - `outputs/tables/general_sentence_models_anchor_count_curve.csv`
  - `outputs/tables/general_sentence_models_corpus_perturbation.csv`

Important caveat: `data/processed/sentence_corpus.jsonl` is not present yet, so this is a sentence-model run on the existing code corpus, not a true sentence-corpus run.

## Main finding

Anchor-relative signatures improved global distance-order preservation, but they did not improve nearest-neighbor MRR in this run.

Across 15 model pairs:

| Metric | Absolute embeddings | Relative signatures | Delta |
|---|---:|---:|---:|
| Mean MRR | 0.8495 | 0.7488 | -0.1007 |
| Mean Spearman rho | 0.3246 | 0.5937 | +0.2691 |
| Mean top-1 overlap | 0.4522 | 0.3122 | -0.1400 |

Relative signatures won on Spearman rho for all 15 pairs, but won on MRR for 0 of 15 pairs.

Interpretation: relative signatures are preserving the broader geometry better, but the 32-anchor signature is not preserving exact local nearest-neighbor identity as well as raw embeddings on this small code corpus.

## Cross-model preservation

| Model A | Model B | Abs MRR | Rel MRR | Delta | Abs rho | Rel rho |
|---|---|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | all-MiniLM-L12-v2 | 0.9402 | 0.8922 | -0.0480 | 0.5392 | 0.7641 |
| all-MiniLM-L6-v2 | paraphrase-MiniLM-L3-v2 | 0.8680 | 0.8353 | -0.0326 | 0.4670 | 0.6562 |
| all-MiniLM-L6-v2 | paraphrase-albert-small-v2 | 0.7320 | 0.6022 | -0.1298 | 0.2053 | 0.3427 |
| all-MiniLM-L6-v2 | e5-small-v2 | 0.8944 | 0.8392 | -0.0552 | 0.3964 | 0.7538 |
| all-MiniLM-L6-v2 | bge-small-en-v1.5 | 0.9125 | 0.8458 | -0.0668 | 0.2764 | 0.6952 |
| all-MiniLM-L12-v2 | paraphrase-MiniLM-L3-v2 | 0.8863 | 0.8306 | -0.0557 | 0.4005 | 0.6352 |
| all-MiniLM-L12-v2 | paraphrase-albert-small-v2 | 0.7600 | 0.5939 | -0.1661 | 0.1797 | 0.3110 |
| all-MiniLM-L12-v2 | e5-small-v2 | 0.9018 | 0.8543 | -0.0475 | 0.3768 | 0.8103 |
| all-MiniLM-L12-v2 | bge-small-en-v1.5 | 0.9156 | 0.8589 | -0.0566 | 0.3252 | 0.7144 |
| paraphrase-MiniLM-L3-v2 | paraphrase-albert-small-v2 | 0.7691 | 0.6217 | -0.1475 | 0.2004 | 0.2649 |
| paraphrase-MiniLM-L3-v2 | e5-small-v2 | 0.9081 | 0.7872 | -0.1210 | 0.3526 | 0.7094 |
| paraphrase-MiniLM-L3-v2 | bge-small-en-v1.5 | 0.8777 | 0.8322 | -0.0455 | 0.3023 | 0.6545 |
| paraphrase-albert-small-v2 | e5-small-v2 | 0.7596 | 0.4979 | -0.2618 | 0.1695 | 0.3627 |
| paraphrase-albert-small-v2 | bge-small-en-v1.5 | 0.7181 | 0.5357 | -0.1824 | 0.3889 | 0.4231 |
| e5-small-v2 | bge-small-en-v1.5 | 0.8997 | 0.8050 | -0.0946 | 0.2887 | 0.8080 |

Best relative-MRR pair:

- `all-MiniLM-L6-v2` vs `all-MiniLM-L12-v2`: relative MRR 0.8922

Worst relative-MRR pair:

- `paraphrase-albert-small-v2` vs `e5-small-v2`: relative MRR 0.4979

Largest Spearman improvements:

- `e5-small-v2` vs `bge-small-en-v1.5`: 0.2887 -> 0.8080
- `all-MiniLM-L12-v2` vs `e5-small-v2`: 0.3768 -> 0.8103
- `all-MiniLM-L6-v2` vs `bge-small-en-v1.5`: 0.2764 -> 0.6952

## Anchor count sweep

Using `all-MiniLM-L6-v2` as the primary model:

| Anchors | MRR vs absolute | Spearman vs absolute | Top-1 overlap |
|---:|---:|---:|---:|
| 8 | 0.8156 | 0.3405 | 0.3167 |
| 16 | 0.8609 | 0.3823 | 0.4583 |
| 32 | 0.9584 | 0.5702 | 0.5667 |

Interpretation: the signature improves sharply as anchors increase from 8 to 32. There is no plateau yet, so this run does not establish the effective anchor count. It says 32 anchors is not obviously too many for this setup.

## Corpus perturbation

| Expansion | Baseline docs | Full docs | Fixed-anchor drift | Refit-anchor drift |
|---:|---:|---:|---:|---:|
| 10% | 60 | 66 | 0.000000 | 0.012629 |
| 25% | 60 | 75 | 0.000000 | 0.012611 |
| 50% | 60 | 90 | 0.000000 | 0.011437 |
| 100% | 60 | 120 | 0.000000 | 0.012012 |

Fixed-anchor drift is zero because deterministic embeddings and fixed anchors make old signatures algebraically unchanged. The relevant number here is refit-anchor drift, which stayed around 0.012. That is low, but this is only 120 documents and should not be treated as conclusive.

## Conclusion

This run partially supports the relative-representation idea:

- Strong support: relative signatures preserved global pairwise ordering better than absolute embeddings across every tested model pair.
- Weak support: relative signatures did not preserve top-k nearest-neighbor MRR better than absolute embeddings.
- Insertion stability is not a problem with fixed anchors in this setup.
- Anchor reselection appears low-drift here, but the corpus is too small to trust that as a production claim.

Next test should use the real sentence corpus once `data/processed/sentence_corpus.jsonl` exists, raise document count beyond 120, and sweep anchors past 32.
