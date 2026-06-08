# LARP Experiment Results

- Run: tenk_minilm_candidate
- Docs sampled: 10000
- Language mix: {'ruby': 1666, 'javascript': 1667, 'java': 1667, 'go': 1666, 'php': 1667, 'python': 1667}
- Models completed: ['sentence-transformers/all-MiniLM-L6-v2']
- Anchor count: 256
- Max tokens/snippet: 160

## Experiment 1: Cross-model Preservation

Not enough completed models for cross-model comparison.

## Experiment 2: Corpus Perturbation

| Expansion | Fixed-anchor drift | Refit-anchor drift |
|---:|---:|---:|

## Experiment 3: Anchor Count

| Anchors | MRR | Spearman | Top-1 overlap |
|---:|---:|---:|---:|

## Experiment 4: Candidate Recall

Mean recall of each model's raw top-10 inside the relative-signature candidate pool.

| Model | Pool | Mean Recall | All Top-k Contained | Any Hit |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 50 | 0.8497 | 0.4070 | 1.0000 |
| all-MiniLM-L6-v2 | 100 | 0.9130 | 0.5730 | 1.0000 |
| all-MiniLM-L6-v2 | 250 | 0.9643 | 0.7890 | 1.0000 |
| all-MiniLM-L6-v2 | 500 | 0.9825 | 0.8810 | 1.0000 |
| all-MiniLM-L6-v2 | 1000 | 0.9925 | 0.9430 | 1.0000 |

## Notes

Fixed-anchor perturbation is algebraically stable in this setup because transformer
embeddings are computed independently per snippet and the anchors are unchanged.
The refit-anchor curve is the meaningful warning signal: if anchors are reselected
when the corpus grows, existing signatures move and a rebuild-like update appears.
