# LARP Experiment Results

- Run: large_minilm_candidate
- Docs sampled: 1000
- Language mix: {'java': 167, 'javascript': 167, 'ruby': 166, 'php': 167, 'go': 166, 'python': 167}
- Models completed: ['sentence-transformers/all-MiniLM-L6-v2']
- Anchor count: 128
- Max tokens/snippet: 160

## Experiment 1: Cross-model Preservation

Not enough completed models for cross-model comparison.

## Experiment 2: Corpus Perturbation

| Expansion | Fixed-anchor drift | Refit-anchor drift |
|---:|---:|---:|
| 10% | 0.000000 | 0.008139 |
| 25% | 0.000000 | 0.007797 |
| 50% | 0.000000 | 0.007619 |
| 100% | 0.000000 | 0.008025 |

## Experiment 3: Anchor Count

| Anchors | MRR | Spearman | Top-1 overlap |
|---:|---:|---:|---:|
| 8 | 0.4562 | 0.4181 | 0.1610 |
| 16 | 0.7248 | 0.5290 | 0.3270 |
| 32 | 0.8621 | 0.5966 | 0.4560 |
| 64 | 0.9149 | 0.6801 | 0.5180 |
| 128 | 0.9615 | 0.7581 | 0.6100 |
| 256 | 0.9715 | 0.8103 | 0.6290 |

## Experiment 4: Candidate Recall

Mean recall of each model's raw top-10 inside the relative-signature candidate pool.

| Model | Pool | Mean Recall | All Top-k Contained | Any Hit |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 25 | 0.8061 | 0.3010 | 1.0000 |
| all-MiniLM-L6-v2 | 50 | 0.8958 | 0.5040 | 1.0000 |
| all-MiniLM-L6-v2 | 100 | 0.9543 | 0.7270 | 1.0000 |
| all-MiniLM-L6-v2 | 250 | 0.9920 | 0.9360 | 1.0000 |
| all-MiniLM-L6-v2 | 500 | 0.9995 | 0.9960 | 1.0000 |

## Notes

Fixed-anchor perturbation is algebraically stable in this setup because transformer
embeddings are computed independently per snippet and the anchors are unchanged.
The refit-anchor curve is the meaningful warning signal: if anchors are reselected
when the corpus grows, existing signatures move and a rebuild-like update appears.
