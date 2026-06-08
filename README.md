# Relative Signature Index Experiments

This repo tests whether fixed anchor-relative signatures can support an incremental semantic search index:

1. Embed documents with a text/code embedding model.
2. Pick a fixed anchor set.
3. Represent each document by its cosine similarities to those anchors.
4. Search the relative-signature space for a candidate pool.
5. Rerank candidates with raw embedding cosine similarity.

The practical claim is not that relative signatures produce the final exact top-k ordering. The practical claim is that they can produce a small candidate pool with high recall, while allowing new documents to be inserted without recomputing existing document signatures.

## Environment

The existing outputs in this checkout were produced with the system Python available on this machine. For reproduction, use a virtual environment.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

GPU is optional. The scripts default to CUDA if PyTorch sees it, otherwise CPU.

## Repo Layout

```text
run_larp_experiment.py          Main experiment runner
scripts/07_signature_index_benchmark.py
                                Relative-index speed benchmark
scripts/00_*.py ... 06_*.py     Earlier staged pipeline scripts
results.md                      Consolidated findings
reports/                        Generated per-run markdown reports
outputs/                        Generated baseline artifacts
experiments/tenk_minilm_candidate/
                                Separate 10k candidate-recall run
data/                           Downloaded and processed local data
```

`data/`, embedding arrays, Python caches, and virtual environments are ignored for future runs. Some generated artifacts were already tracked before cleanup; see the cleanup note below.

## Reproduce Key Runs

Small code-model run:

```powershell
python .\run_larp_experiment.py `
  --model-suite code `
  --n-docs 120 `
  --anchor-count 32 `
  --batch-size 8 `
  --max-tokens 160 `
  --run-name code_models_candidate `
  --artifact-root experiments\code_models_candidate
```

General sentence-model suite:

```powershell
python .\run_larp_experiment.py `
  --model-suite general-sentence `
  --n-docs 120 `
  --anchor-count 32 `
  --batch-size 8 `
  --max-tokens 160 `
  --run-name general_sentence_candidate `
  --artifact-root experiments\general_sentence_candidate `
  --candidate-pools 10 25 50 75 100
```

1k candidate-recall run:

```powershell
python .\run_larp_experiment.py `
  --n-docs 1000 `
  --anchor-count 128 `
  --batch-size 16 `
  --max-tokens 160 `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --run-name large_minilm_candidate `
  --artifact-root experiments\large_minilm_candidate `
  --candidate-pools 25 50 100 250 500
```

10k candidate-recall run in a separate artifact folder:

```powershell
python .\run_larp_experiment.py `
  --n-docs 10000 `
  --anchor-count 256 `
  --batch-size 32 `
  --max-tokens 160 `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --run-name tenk_minilm_candidate `
  --artifact-root experiments\tenk_minilm_candidate `
  --skip-cross-model `
  --skip-anchor-count `
  --skip-perturbation `
  --sample-queries 1000 `
  --candidate-pools 50 100 250 500 1000
```

10k speed benchmark:

```powershell
python .\scripts\07_signature_index_benchmark.py
```

## Current Key Result

On the 10k MiniLM run with 256 fixed anchors and 1,000 sampled queries, relative-signature candidate pools recovered the raw embedding top-10 at:

| Candidate Pool | Mean Recall of Raw Top-10 |
|---:|---:|
| 50 | 0.8497 |
| 100 | 0.9130 |
| 250 | 0.9643 |
| 500 | 0.9825 |
| 1000 | 0.9925 |

At 10k documents, exact vectorized search over relative signatures was faster than a Python tree prototype:

| Method | Query Time |
|---|---:|
| Vectorized exact relative search | ~0.084 ms/query |
| Random-projection forest prototype | ~4.92 ms/query |

So for 10k, brute vectorized relative search is already sufficient. At larger scale, use a compiled ANN index such as HNSW/FAISS/ScaNN rather than a Python object tree.

## Cleanup Note

The repo already had generated artifacts tracked before this cleanup. `.gitignore` prevents future generated data from being added, but it does not untrack files already in git.

To untrack generated files while keeping them on disk, run only if you are ready for that git index change:

```powershell
git rm --cached -r data outputs __pycache__ scripts\__pycache__
git add .gitignore README.md requirements.txt results.md reports scripts run_larp_experiment.py config.yaml
```

Review with:

```powershell
git status --short
```
