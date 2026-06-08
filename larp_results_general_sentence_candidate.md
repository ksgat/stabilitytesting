# LARP Experiment Results

- Run: general_sentence_candidate
- Docs sampled: 120
- Language mix: {'java': 20, 'php': 20, 'go': 20, 'ruby': 20, 'javascript': 20, 'python': 20}
- Models completed: ['sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/paraphrase-MiniLM-L3-v2', 'sentence-transformers/paraphrase-albert-small-v2', 'intfloat/e5-small-v2', 'BAAI/bge-small-en-v1.5']
- Anchor count: 32
- Max tokens/snippet: 160

## Experiment 1: Cross-model Preservation

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

## Experiment 2: Corpus Perturbation

| Expansion | Fixed-anchor drift | Refit-anchor drift |
|---:|---:|---:|
| 10% | 0.000000 | 0.012629 |
| 25% | 0.000000 | 0.012611 |
| 50% | 0.000000 | 0.011437 |
| 100% | 0.000000 | 0.012012 |

## Experiment 3: Anchor Count

| Anchors | MRR | Spearman | Top-1 overlap |
|---:|---:|---:|---:|
| 8 | 0.8156 | 0.3405 | 0.3167 |
| 16 | 0.8609 | 0.3823 | 0.4583 |
| 32 | 0.9584 | 0.5702 | 0.5667 |

## Experiment 4: Candidate Recall

Mean recall of each model's raw top-10 inside the relative-signature candidate pool.

| Model | Pool | Mean Recall | All Top-k Contained | Any Hit |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 10 | 0.5792 | 0.0250 | 1.0000 |
| all-MiniLM-L6-v2 | 25 | 0.7842 | 0.2583 | 1.0000 |
| all-MiniLM-L6-v2 | 50 | 0.9533 | 0.7667 | 1.0000 |
| all-MiniLM-L6-v2 | 75 | 0.9842 | 0.9167 | 1.0000 |
| all-MiniLM-L6-v2 | 100 | 1.0000 | 1.0000 | 1.0000 |
| all-MiniLM-L12-v2 | 10 | 0.5600 | 0.0000 | 1.0000 |
| all-MiniLM-L12-v2 | 25 | 0.7783 | 0.1833 | 1.0000 |
| all-MiniLM-L12-v2 | 50 | 0.9233 | 0.5250 | 1.0000 |
| all-MiniLM-L12-v2 | 75 | 0.9900 | 0.9000 | 1.0000 |
| all-MiniLM-L12-v2 | 100 | 1.0000 | 1.0000 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 10 | 0.5750 | 0.0000 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 25 | 0.8292 | 0.3000 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 50 | 0.9567 | 0.7000 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 75 | 0.9858 | 0.8833 | 1.0000 |
| paraphrase-MiniLM-L3-v2 | 100 | 0.9983 | 0.9833 | 1.0000 |
| paraphrase-albert-small-v2 | 10 | 0.5867 | 0.0083 | 1.0000 |
| paraphrase-albert-small-v2 | 25 | 0.8975 | 0.3833 | 1.0000 |
| paraphrase-albert-small-v2 | 50 | 0.9942 | 0.9500 | 1.0000 |
| paraphrase-albert-small-v2 | 75 | 0.9992 | 0.9917 | 1.0000 |
| paraphrase-albert-small-v2 | 100 | 1.0000 | 1.0000 | 1.0000 |
| e5-small-v2 | 10 | 0.4792 | 0.0000 | 0.9917 |
| e5-small-v2 | 25 | 0.7025 | 0.1917 | 0.9917 |
| e5-small-v2 | 50 | 0.8450 | 0.3500 | 1.0000 |
| e5-small-v2 | 75 | 0.9450 | 0.6667 | 1.0000 |
| e5-small-v2 | 100 | 0.9867 | 0.8750 | 1.0000 |
| bge-small-en-v1.5 | 10 | 0.5642 | 0.0000 | 1.0000 |
| bge-small-en-v1.5 | 25 | 0.8042 | 0.3000 | 1.0000 |
| bge-small-en-v1.5 | 50 | 0.9375 | 0.6250 | 1.0000 |
| bge-small-en-v1.5 | 75 | 0.9850 | 0.8833 | 1.0000 |
| bge-small-en-v1.5 | 100 | 1.0000 | 1.0000 | 1.0000 |

## Notes

Fixed-anchor perturbation is algebraically stable in this setup because transformer
embeddings are computed independently per snippet and the anchors are unchanged.
The refit-anchor curve is the meaningful warning signal: if anchors are reselected
when the corpus grows, existing signatures move and a rebuild-like update appears.
