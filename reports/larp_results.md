# LARP Experiment Results

- Docs sampled: 120
- Language mix: {'java': 20, 'php': 20, 'go': 20, 'ruby': 20, 'javascript': 20, 'python': 20}
- Models completed: ['microsoft/codebert-base', 'Salesforce/codet5-small', 'microsoft/unixcoder-base']
- Anchor count: 32
- Max tokens/snippet: 160

## Experiment 1: Cross-model Preservation

| Model A | Model B | Abs MRR | Rel MRR | Delta | Abs rho | Rel rho |
|---|---|---:|---:|---:|---:|---:|
| codebert-base | codet5-small | 0.8669 | 0.6171 | -0.2498 | 0.7436 | 0.7116 |
| codebert-base | unixcoder-base | 0.7513 | 0.5154 | -0.2359 | 0.3242 | 0.4112 |
| codet5-small | unixcoder-base | 0.7825 | 0.5745 | -0.2079 | 0.3725 | 0.5155 |

## Experiment 2: Corpus Perturbation

| Expansion | Fixed-anchor drift | Refit-anchor drift |
|---:|---:|---:|
| 10% | 0.000000 | 0.000386 |
| 25% | 0.000000 | 0.000505 |
| 50% | 0.000000 | 0.000418 |
| 100% | 0.000000 | 0.000637 |

## Experiment 3: Anchor Count

| Anchors | MRR | Spearman | Top-1 overlap |
|---:|---:|---:|---:|
| 8 | 0.8034 | 0.9035 | 0.2500 |
| 16 | 0.9260 | 0.9182 | 0.2667 |
| 32 | 0.9454 | 0.9235 | 0.3167 |

## Notes

Fixed-anchor perturbation is algebraically stable in this setup because transformer
embeddings are computed independently per snippet and the anchors are unchanged.
The refit-anchor curve is the meaningful warning signal: if anchors are reselected
when the corpus grows, existing signatures move and a rebuild-like update appears.
