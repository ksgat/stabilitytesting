# LARP Experiment Results

- Run: robustness_3k_general_rerun
- Docs sampled: 3000
- Language mix: {'java': 506, 'go': 482, 'python': 505, 'php': 490, 'ruby': 505, 'javascript': 512}
- Models completed: ['sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/paraphrase-MiniLM-L3-v2', 'sentence-transformers/paraphrase-albert-small-v2', 'intfloat/e5-small-v2', 'BAAI/bge-small-en-v1.5']
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
| all-MiniLM-L6-v2 | 50 | 0.8671 | 0.3530 | 1.0000 |
| all-MiniLM-L6-v2 | 100 | 0.9335 | 0.5960 | 1.0000 |
| all-MiniLM-L6-v2 | 250 | 0.9796 | 0.8430 | 1.0000 |
| all-MiniLM-L6-v2 | 500 | 0.9952 | 0.9590 | 1.0000 |
| all-MiniLM-L6-v2 | 1000 | 0.9992 | 0.9920 | 1.0000 |
| all-MiniLM-L12-v2 | 50 | 0.8246 | 0.2540 | 1.0000 |
| all-MiniLM-L12-v2 | 100 | 0.9087 | 0.4770 | 1.0000 |
| all-MiniLM-L12-v2 | 250 | 0.9703 | 0.7800 | 1.0000 |
| all-MiniLM-L12-v2 | 500 | 0.9908 | 0.9140 | 1.0000 |
| all-MiniLM-L12-v2 | 1000 | 0.9987 | 0.9880 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 50 | 0.8565 | 0.3340 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 100 | 0.9307 | 0.5870 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 250 | 0.9786 | 0.8440 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 500 | 0.9945 | 0.9530 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 1000 | 0.9989 | 0.9890 | 1.0000 |
| paraphrase-albert-small-v2 | 50 | 0.5522 | 0.0120 | 0.9980 |
| paraphrase-albert-small-v2 | 100 | 0.6815 | 0.0640 | 1.0000 |
| paraphrase-albert-small-v2 | 250 | 0.8460 | 0.3140 | 1.0000 |
| paraphrase-albert-small-v2 | 500 | 0.9395 | 0.6150 | 1.0000 |
| paraphrase-albert-small-v2 | 1000 | 0.9900 | 0.9190 | 1.0000 |
| e5-small-v2 | 50 | 0.8074 | 0.2510 | 1.0000 |
| e5-small-v2 | 100 | 0.8994 | 0.4910 | 1.0000 |
| e5-small-v2 | 250 | 0.9586 | 0.7270 | 1.0000 |
| e5-small-v2 | 500 | 0.9802 | 0.8570 | 1.0000 |
| e5-small-v2 | 1000 | 0.9897 | 0.9200 | 1.0000 |
| bge-small-en-v1.5 | 50 | 0.7892 | 0.1900 | 1.0000 |
| bge-small-en-v1.5 | 100 | 0.8826 | 0.3900 | 1.0000 |
| bge-small-en-v1.5 | 250 | 0.9606 | 0.7240 | 1.0000 |
| bge-small-en-v1.5 | 500 | 0.9869 | 0.9050 | 1.0000 |
| bge-small-en-v1.5 | 1000 | 0.9985 | 0.9860 | 1.0000 |

## Notes

Fixed-anchor perturbation is algebraically stable in this setup because transformer
embeddings are computed independently per snippet and the anchors are unchanged.
The refit-anchor curve is the meaningful warning signal: if anchors are reselected
when the corpus grows, existing signatures move and a rebuild-like update appears.
