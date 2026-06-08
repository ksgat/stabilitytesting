Additional possible route to explore

## The problem

Semantic search has no clean solution in its truest form: given a query and a corpus, find semantically similar documents with no pre-built index, no pre-computation, nothing.

Every approach collapses into one of three failures: it is slow, it is approximate, or it requires offline work that defeats the point. The underlying reason is not just compute or algorithms. Meaning has no axiom system. B-trees work because `5 < 10` is true forever regardless of what other numbers exist. Semantic distance is relational, context-dependent, and has no total ordering to build a hierarchy on.

## The observation

Absolute embedding positions are essentially meaningless across models: Moschella et al. report near-zero cross-model MRR for absolute representations. Two independently trained models produce coordinate systems that are random relative to each other.

Anchor-relative signatures represent each document by its pattern of distances to a fixed set of landmark points. Those signatures preserve structure much better across models. The relative structure is the stable thing; absolute coordinates are mostly model-local.

## The hypothesis

If anchor-relative signatures remain useful under corpus updates, then an incremental semantic index becomes more plausible. The write path can compute or update relative signatures, and the query path can use an already-current structure instead of triggering a full rebuild.

The load-bearing empirical question is not just "do signatures drift?" For deterministic embedding models and fixed anchors, old signatures do not drift when new documents are added. The useful questions are:

- Do anchor-relative signatures preserve neighbor structure better than raw embedding neighborhoods across model families?
- How many anchors are needed before signatures become discriminative?
- When new documents are inserted, do old documents' top-k neighborhoods change only locally?
- If anchors are reselected after the corpus grows, how unstable are rankings?

## Data contract

The downloader agent should produce:

```text
data/corpus.jsonl
```

Each row:

```json
{"id":"unique-stable-id","text":"code or document text","path":"optional/source/path.py","lang":"optional-language","repo":"optional-repo"}
```

Rules:

- `id` must be stable across runs.
- `text` must be non-empty.
- First-pass documents should be roughly 50 to 4000 characters.
- Exact duplicate `text` values should be removed or allowed for `00_validate_corpus.py` to report.
- The scripts do not download data. They only validate, embed, cache, compare, and report.

## Files added for testing

```text
config.yaml
requirements.txt
scripts/00_validate_corpus.py
scripts/01_embed_models.py
scripts/02_make_signatures.py
scripts/03_cross_model_preservation.py
scripts/04_anchor_count_sweep.py
scripts/05_incremental_insert_test.py
scripts/06_report.py
```

## Run order

```bash
python scripts/00_validate_corpus.py --corpus data/corpus.jsonl
python scripts/01_embed_models.py --config config.yaml
python scripts/02_make_signatures.py --config config.yaml
python scripts/03_cross_model_preservation.py --config config.yaml
python scripts/04_anchor_count_sweep.py --config config.yaml
python scripts/05_incremental_insert_test.py --config config.yaml
python scripts/06_report.py --run-dir runs/baseline
```

## What each script does

| Script | Purpose |
|---|---|
| `00_validate_corpus.py` | Checks JSONL shape, duplicate ids, duplicate texts, length filters, and language mix. |
| `01_embed_models.py` | Embeds the same corpus with each configured model and caches `.npy` arrays. |
| `02_make_signatures.py` | Picks stable anchor ids and creates anchor-relative signatures for every model. |
| `03_cross_model_preservation.py` | Compares raw embedding neighbor preservation against anchor-relative signature preservation across model pairs. |
| `04_anchor_count_sweep.py` | Measures how retrieval preservation changes as anchor count increases. |
| `05_incremental_insert_test.py` | Simulates corpus growth and measures old-document neighbor disruption plus anchor reselection instability. |
| `06_report.py` | Writes a markdown report from metric JSON files. |

## First-pass decision thresholds

Treat these as rough gates, not final claims:

| Test | Promising | Weak |
|---|---:|---:|
| Cross-model relative MRR | `> 0.70` | `< 0.40` |
| Relative vs raw MRR lift | `+0.20` or better | `< +0.05` |
| Anchor knee | `<= 128 anchors` | `>= 512 anchors with no plateau` |
| Retained old top-k after inserts | `> 0.80` | `< 0.50` |
| Reselected-anchor MRR | `> 0.80` | `< 0.50` |

The first serious result to look for is not whether retrieval quality is high in absolute terms. It is whether anchor-relative signatures preserve structure better than raw embedding neighborhoods across model families.
