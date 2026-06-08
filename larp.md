# Semantic Relationship Preservation: Testing Moschella's Hypothesis on Code Embeddings

## Problem Statement

Semantic search over code is unsolved in its truest form. The goal — given a query and a corpus, find semantically similar code with no pre-built index, no pre-computation, nothing — collides with a fundamental wall:

**Embedding is expensive. Comparison is expensive. And you need both to understand meaning computationally.**

Every known solution trades one of three properties:

- **Fast** — pre-built indexes, but then it's not JIT
- **Cheap** — approximations, but then recall degrades
- **Good** — full semantic understanding, but then it's slow and expensive

The hypothesis being explored here is whether **relative relationships between embeddings** are stable enough across different models and corpus changes to serve as the foundation for a new kind of index — one that builds incrementally as code changes, never requiring a full batch rebuild.

The key insight: traditional indexes (B-trees) work because numeric ordering is eternal. `5 < 10` is true forever regardless of what other numbers exist. Absolute embedding positions don't have this property — add a new document and theoretically every position shifts. But **relative relationships** between embeddings might be conservative in the way numeric ordering is, making incremental indexing viable.

---

## Background: Moschella et al.

**Paper**: "Relative Representations Enable Zero-Shot Latent Space Communication" (Moschella et al., 2022)

**Core claim**: When you represent items not by their absolute position in embedding space, but by their similarity/distance to a fixed set of anchor points, these relative representations are preserved across different embedding models trained on the same data.

**Key result**: Absolute cross-model similarity: ~0.00 MRR. Anchor-relative similarity: 0.94–0.98 MRR.

**What this was tested on**: Word embeddings (Word2Vec, FastText) — static, context-free, relatively similar architectures.

**What has NOT been tested**: Modern contextual models (BERT-family, code-specific models), across significantly different architectures, on code specifically, under corpus perturbation.

---

## What You Are Testing

**Central question**: Does Moschella's result generalize to modern code embedding models?

Specifically:

1. Do nearest-neighbor rank orderings survive across different code embedding models?
2. Does relative structure (anchor-relative signatures) preserve better than absolute structure?
3. How does preservation degrade as you add new code to the corpus?
4. Is the dimensionality of the relative signature space meaningfully lower than raw embedding space?

If preservation is high: there is a principled foundation for an incremental semantic index over code.
If preservation is low: the approach needs rethinking before any system is built on it.

---

## Setup

### Hardware
- Runs fine on a 3050 (8GB VRAM)
- Stick to sub-1B parameter embedding models
- 5000 files is sufficient for statistical significance

### Models to Test
```python
MODELS = [
    "microsoft/codebert-base",           # 125M, RoBERTa-based
    "Salesforce/codet5p-110m-embedding", # 110M, encoder-decoder derived
    "microsoft/unixcoder-base",          # 125M, unified cross-modal
]
```

Three models from different training approaches but similar parameter counts. The goal is architectural diversity, not scale.

### Corpus
Pull from any of:
- Your own codebase
- GitHub repos via `datasets` library (`codeparrot/github-code`)
- The Stack (HuggingFace)

Target: ~5000 code files, mixed languages, mixed function sizes. Diversity matters more than volume here.

---

## Experiment 1: Absolute vs Relative Preservation

**What**: Measure how well nearest-neighbor structure survives across model pairs, comparing absolute embeddings vs anchor-relative signatures.

**Metric**: Spearman rank correlation of pairwise distance matrices. Also MRR (Mean Reciprocal Rank) of nearest neighbors.

```python
import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

def embed_corpus(corpus, model_name):
    model = SentenceTransformer(model_name)
    return model.encode(corpus, batch_size=32, show_progress_bar=True)

def pairwise_distances(embeddings):
    # cosine distance matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    similarities = normalized @ normalized.T
    return 1 - similarities  # distance not similarity

def anchor_relative_signature(embeddings, anchor_embeddings):
    # represent each doc as its cosine similarity to each anchor
    norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_a = np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
    e_norm = embeddings / norms_e
    a_norm = anchor_embeddings / norms_a
    return e_norm @ a_norm.T  # shape: (n_docs, n_anchors)

def mrr_preservation(dist_a, dist_b, k=10):
    """How well does model_a's top-k predict model_b's top-k"""
    n = dist_a.shape[0]
    reciprocal_ranks = []
    for i in range(n):
        top_k_a = np.argsort(dist_a[i])[1:k+1]  # exclude self
        top_k_b = np.argsort(dist_b[i])[1:k+1]
        for rank, idx in enumerate(top_k_b):
            if idx in top_k_a:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)

# Run experiment
corpus = load_your_corpus(n=5000)

# select anchors — random subset for now (this is a known weakness, see notes)
anchor_indices = np.random.choice(len(corpus), size=512, replace=False)
anchors = [corpus[i] for i in anchor_indices]

results = {}
for model_name in MODELS:
    embeddings = embed_corpus(corpus, model_name)
    anchor_embeddings = embeddings[anchor_indices]
    signatures = anchor_relative_signature(embeddings, anchor_embeddings)
    
    results[model_name] = {
        "embeddings": embeddings,
        "signatures": signatures,
        "dist_matrix": pairwise_distances(embeddings),
        "sig_dist_matrix": pairwise_distances(signatures),
    }

# Cross-model preservation
model_pairs = [
    ("microsoft/codebert-base", "Salesforce/codet5p-110m-embedding"),
    ("microsoft/codebert-base", "microsoft/unixcoder-base"),
    ("Salesforce/codet5p-110m-embedding", "microsoft/unixcoder-base"),
]

for m_a, m_b in model_pairs:
    # absolute
    abs_mrr = mrr_preservation(results[m_a]["dist_matrix"], results[m_b]["dist_matrix"])
    # relative
    rel_mrr = mrr_preservation(results[m_a]["sig_dist_matrix"], results[m_b]["sig_dist_matrix"])
    
    print(f"{m_a.split('/')[-1]} vs {m_b.split('/')[-1]}")
    print(f"  Absolute MRR: {abs_mrr:.4f}")
    print(f"  Relative MRR: {rel_mrr:.4f}")
    print(f"  Delta: {rel_mrr - abs_mrr:+.4f}")
```

**What you want to see**: Relative MRR significantly higher than Absolute MRR, ideally replicating Moschella's 0.94+ result. If the delta is small, the hypothesis is in trouble.

---

## Experiment 2: Corpus Perturbation Stability

**What**: Add new documents incrementally. Measure how much the relative signatures of existing documents shift.

**Why**: This is the load-bearing claim for the incremental index idea. If adding new files destabilizes existing signatures, you need a full rebuild on every commit. If signatures are stable, you only update locally.

```python
def signature_drift(original_sigs, perturbed_sigs):
    """Cosine similarity between original and perturbed signatures for same docs"""
    orig_norm = original_sigs / np.linalg.norm(original_sigs, axis=1, keepdims=True)
    pert_norm = perturbed_sigs / np.linalg.norm(perturbed_sigs, axis=1, keepdims=True)
    return np.diag(orig_norm @ pert_norm.T).mean()

# baseline: embed 4000 docs
baseline_corpus = corpus[:4000]
baseline_embeddings = embed_corpus(baseline_corpus, MODELS[0])
baseline_anchors = baseline_embeddings[anchor_indices[:512]]
baseline_sigs = anchor_relative_signature(baseline_embeddings, baseline_anchors)

# perturb: add 1000 new docs
full_embeddings = embed_corpus(corpus, MODELS[0])
full_anchors = full_embeddings[anchor_indices[:512]]
full_sigs = anchor_relative_signature(full_embeddings[:4000], full_anchors)

drift = signature_drift(baseline_sigs, full_sigs)
print(f"Signature drift after 25% corpus expansion: {1 - drift:.4f}")
# close to 0 = stable, close to 1 = completely unstable
```

Run this at multiple expansion levels: +10%, +25%, +50%, +100%. Plot the drift curve.

---

## Experiment 3: Anchor Count vs Retrieval Quality

**What**: How many anchors do you actually need before signatures become discriminative?

**Why**: If 32 anchors work as well as 512, you've found that relative semantic space is much lower dimensional than raw embedding space. That's a significant result on its own.

```python
anchor_counts = [16, 32, 64, 128, 256, 512]
baseline_mrr = mrr_preservation(
    results[MODELS[0]]["dist_matrix"],
    results[MODELS[0]]["dist_matrix"]
)  # ceiling: perfect retrieval against itself

for n_anchors in anchor_counts:
    selected_anchors = results[MODELS[0]]["embeddings"][anchor_indices[:n_anchors]]
    sigs = anchor_relative_signature(
        results[MODELS[0]]["embeddings"], 
        selected_anchors
    )
    sig_dists = pairwise_distances(sigs)
    mrr = mrr_preservation(results[MODELS[0]]["dist_matrix"], sig_dists)
    print(f"Anchors: {n_anchors:4d} | MRR: {mrr:.4f}")
```

Plot the curve. Look for the knee — where adding more anchors stops helping. That's your effective dimensionality of relative semantic space.

---

## What to Report

For each experiment, you want:

| Experiment | Key Metric | Good Result | Bad Result |
|---|---|---|---|
| 1: Cross-model | Relative MRR vs Absolute MRR | Relative >> Absolute | Similar or Relative < Absolute |
| 2: Perturbation | Signature drift at +25% corpus | < 0.05 drift | > 0.2 drift |
| 3: Anchor count | Knee of MRR curve | Knee at < 128 anchors | No clear knee, needs 512+ |

---

## Known Weaknesses in This Experiment

**Anchor selection is random**: This is a placeholder. Random anchors are probably not optimal. The experiment measures whether the hypothesis is alive at all — not whether it's maximally exploited. If results are promising, anchor selection becomes the next problem.

**Same-model anchors**: Anchors are selected from model A's embeddings and used to test model A. Cross-model anchor transfer (anchors from model A used in model B's space) is a harder test and probably the right one eventually.

**No ground truth**: There's no labeled dataset of "these two code snippets are semantically equivalent." MRR against the same model is a proxy, not a ground truth. This is a known limitation of working in an unsupervised setting.

**Context-dependence**: These models produce fixed embeddings per snippet. In production, snippets have context (surrounding code, imports, call sites). That context changes the embedding and might destabilize relative signatures in ways this experiment won't catch.

---

## The Actual Question This Answers

Not "does semantic code search work."

**"Is relative structure stable enough across models and corpus changes to be worth building on."**

If yes: the incremental index idea has a foundation and the H100 weekend is about building the system.

If no: you've saved yourself from building on sand, and the problem needs a different primitive entirely.

Either result is useful. Run the experiment before building anything.



