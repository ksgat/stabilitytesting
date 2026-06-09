# LARP Model Robustness Rerun

## Scope

- Embedding files evaluated: 5
- Documents per embedding file: 3000
- Sampled queries per model: 1000
- Anchor count: 256
- Raw truth target: top-10 nearest neighbors in each model's own embedding space
- Candidate pools: 50, 100, 250, 500, 1000

## Method

Each model is evaluated independently. Raw embeddings are L2-normalized, raw top-k neighbors are computed by cosine similarity, and LARP signatures are built by comparing every document to a fixed anchor set. The measured search step ranks documents by cosine similarity between relative signatures, then reports how many true raw top-k neighbors appear inside the relative candidate pool.

Strategies tested:

- `random_raw`: random anchors with L2-normalized raw anchor similarities.
- `random_row_zscore`: random anchors with each document signature centered and scaled across anchors.
- `farthest_raw`: farthest-point anchors with raw normalized signatures.
- `farthest_row_zscore`: farthest-point anchors with row z-score signatures. This is the current best setting from the 10k anchor ablation.

This is still a two-stage search test: the relative signature is evaluated as candidate generation, not as a final replacement for raw cosine reranking.

## Best Configuration Results

| model | dim | docs | pool 50 | pool 100 | pool 250 | pool 500 | pool 1000 | ms/query @ pool 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intfloat/e5-small-v2 | 384 | 3000 | 0.8486 | 0.9349 | 0.9857 | 0.9974 | 0.9996 | 1.043 |
| sentence-transformers/all-MiniLM-L12-v2 | 384 | 3000 | 0.8729 | 0.9355 | 0.9824 | 0.9962 | 0.9998 | 0.771 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 3000 | 0.9015 | 0.9550 | 0.9889 | 0.9983 | 0.9999 | 0.879 |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | 384 | 3000 | 0.8962 | 0.9596 | 0.9935 | 0.9993 | 1.0000 | 0.755 |
| sentence-transformers/paraphrase-albert-small-v2 | 768 | 3000 | 0.6856 | 0.7908 | 0.9008 | 0.9625 | 0.9921 | 0.785 |

## Strategy Robustness at Pool 250

| strategy | mean recall | min recall | max recall |
| --- | --- | --- | --- |
| farthest_raw | 0.9636 | 0.8753 | 0.9887 |
| farthest_row_zscore | 0.9703 | 0.9008 | 0.9935 |
| random_raw | 0.9438 | 0.8414 | 0.9810 |
| random_row_zscore | 0.9572 | 0.8708 | 0.9851 |

## Findings

- The robustness question is not whether signatures exactly replace embedding search; they do not yet. The useful behavior is whether a small, fixed relative vector reliably catches the raw neighbors as a candidate generator.
- `farthest_row_zscore` is the main candidate for a stable growing index because anchor choice is fixed after build and new documents only need one embedding plus anchor comparisons.
- If the best setting is consistently high across models at pool 250-500, the idea is probably not a single-model fluke. If one model collapses, the index needs model-specific calibration before it can be called universal.
- The compute profile is favorable at this scale: signature search is matrix multiplication over `docs x anchors`, and insert cost is one row of raw embedding plus one row of anchor similarities.

## Generated Artifacts

- CSV: `experiments/robustness_3k_general/model_robustness/model_robustness.csv`
- Best-config plot: `experiments/robustness_3k_general/model_robustness/model_robustness_curves.png`
- Strategy plot: `experiments/robustness_3k_general/model_robustness/model_robustness_strategy_pool250.png`
