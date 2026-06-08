import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr


def slug(name):
    return name.replace("/", "__")


def normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def cosine_distance_matrix(x):
    x = normalize(x.astype("float32"))
    return 1.0 - (x @ x.T)


def topk_indices(dist, k):
    n = dist.shape[0]
    k = min(k, n - 1)
    order = np.argpartition(dist, kth=k, axis=1)[:, :k + 1]
    result = []
    for i, candidates in enumerate(order):
        candidates = candidates[candidates != i]
        candidates = candidates[np.argsort(dist[i, candidates])][:k]
        result.append(candidates)
    return result


def mrr_agreement(reference_dist, candidate_dist, k):
    reference_topk = topk_indices(reference_dist, k)
    scores = []

    for i, ref_neighbors in enumerate(reference_topk):
        ref_set = set(ref_neighbors.tolist())
        ranked = np.argsort(candidate_dist[i])
        rank_without_self = 0
        score = 0.0

        for idx in ranked:
            if idx == i:
                continue
            rank_without_self += 1
            if idx in ref_set:
                score = 1.0 / rank_without_self
                break
            if rank_without_self > 1000:
                break

        scores.append(score)

    return float(np.mean(scores))


def topk_overlap(dist_a, dist_b, k):
    a = topk_indices(dist_a, k)
    b = topk_indices(dist_b, k)
    overlaps = []
    for ai, bi in zip(a, b):
        overlaps.append(len(set(ai.tolist()) & set(bi.tolist())) / len(ai))
    return float(np.mean(overlaps))


def sampled_spearman(dist_a, dist_b, sample_pairs, seed):
    n = dist_a.shape[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)
    keep = i != j
    coef, _ = spearmanr(dist_a[i[keep], j[keep]], dist_b[i[keep], j[keep]])
    return float(coef)


def compare_pair(name_a, matrix_a, name_b, matrix_b, k, sample_pairs, seed):
    return {
        "a": name_a,
        "b": name_b,
        "mrr_a_to_b": mrr_agreement(matrix_a, matrix_b, k),
        "mrr_b_to_a": mrr_agreement(matrix_b, matrix_a, k),
        "topk_overlap": topk_overlap(matrix_a, matrix_b, k),
        "sampled_spearman": sampled_spearman(matrix_a, matrix_b, sample_pairs, seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(config["run_dir"])
    metric_dir = run_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    k = config["metrics"]["k"]
    sample_pairs = config["metrics"]["spearman_sample_pairs"]
    model_names = [model["name"] for model in config["models"]]

    raw_dist = {}
    sig_dist = {}
    for model_name in model_names:
        raw = np.load(run_dir / "embeddings" / f"{slug(model_name)}.npy")
        sig = np.load(run_dir / "signatures" / f"{slug(model_name)}.npy")
        raw_dist[model_name] = cosine_distance_matrix(raw)
        sig_dist[model_name] = cosine_distance_matrix(sig)

    rows = []
    for a, b in itertools.combinations(model_names, 2):
        raw_metrics = compare_pair(a, raw_dist[a], b, raw_dist[b], k, sample_pairs, config["seed"])
        sig_metrics = compare_pair(a, sig_dist[a], b, sig_dist[b], k, sample_pairs, config["seed"])
        rows.append({"representation": "raw_embedding_neighbors", **raw_metrics})
        rows.append({"representation": "anchor_relative_signature_neighbors", **sig_metrics})

    out_path = metric_dir / "cross_model_preservation.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
