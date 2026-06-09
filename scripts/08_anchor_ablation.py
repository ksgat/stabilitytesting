import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, kth=k - 1)[:k]
    return part[np.argsort(-scores[part])]


def load_languages(corpus_path: Path, n_docs: int) -> np.ndarray:
    if not corpus_path.exists():
        return np.array(["unknown"] * n_docs)
    langs = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            langs.append(row.get("language", "unknown"))
            if len(langs) >= n_docs:
                break
    if len(langs) < n_docs:
        langs.extend(["unknown"] * (n_docs - len(langs)))
    return np.array(langs[:n_docs])


def unique_fill(selected: list[int], n_docs: int, count: int, rng: np.random.Generator) -> np.ndarray:
    seen = set()
    out = []
    for idx in selected:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        if len(out) >= count:
            return np.array(out, dtype=np.int32)
    for idx in rng.permutation(n_docs):
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        if len(out) >= count:
            break
    return np.array(out, dtype=np.int32)


def random_anchors(raw: np.ndarray, count: int, rng: np.random.Generator, **_) -> np.ndarray:
    return rng.choice(len(raw), size=count, replace=False).astype(np.int32)


def language_stratified_random(raw: np.ndarray, count: int, rng: np.random.Generator, languages: np.ndarray, **_) -> np.ndarray:
    selected = []
    labels = sorted(set(languages))
    per_lang = max(1, count // max(1, len(labels)))
    for label in labels:
        candidates = np.flatnonzero(languages == label)
        if len(candidates):
            selected.extend(rng.choice(candidates, size=min(per_lang, len(candidates)), replace=False))
    return unique_fill(selected, len(raw), count, rng)


def farthest_point(raw: np.ndarray, count: int, rng: np.random.Generator, start: str = "random", **_) -> np.ndarray:
    if start == "centroid":
        centroid = l2_normalize(raw.mean(axis=0, keepdims=True))[0]
        first = int(np.argmax(raw @ centroid))
    else:
        first = int(rng.integers(0, len(raw)))
    selected = [first]
    max_sim = raw @ raw[first]
    for _ in range(1, count):
        next_idx = int(np.argmin(max_sim))
        selected.append(next_idx)
        max_sim = np.maximum(max_sim, raw @ raw[next_idx])
    return np.array(selected, dtype=np.int32)


def pca_extremes(raw: np.ndarray, count: int, rng: np.random.Generator, n_components: int = 32, **_) -> np.ndarray:
    pcs = PCA(n_components=min(n_components, raw.shape[1]), random_state=0).fit_transform(raw)
    selected = []
    per_pc = max(1, count // (2 * pcs.shape[1]))
    for col in range(pcs.shape[1]):
        order = np.argsort(pcs[:, col])
        selected.extend(order[:per_pc])
        selected.extend(order[-per_pc:])
    return unique_fill(selected, len(raw), count, rng)


def kmeans_medoid(raw: np.ndarray, count: int, rng: np.random.Generator, **_) -> np.ndarray:
    km = MiniBatchKMeans(
        n_clusters=count,
        random_state=int(rng.integers(0, 2**31 - 1)),
        batch_size=2048,
        n_init=3,
        max_iter=100,
    )
    labels = km.fit_predict(raw)
    centers = l2_normalize(km.cluster_centers_.astype(np.float32))
    selected = []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        if len(members) == 0:
            continue
        selected.append(members[int(np.argmax(raw[members] @ centers[cluster]))])
    return unique_fill(selected, len(raw), count, rng)


def kmeans_boundary(raw: np.ndarray, count: int, rng: np.random.Generator, **_) -> np.ndarray:
    km = MiniBatchKMeans(
        n_clusters=count,
        random_state=int(rng.integers(0, 2**31 - 1)),
        batch_size=2048,
        n_init=3,
        max_iter=100,
    )
    labels = km.fit_predict(raw)
    centers = l2_normalize(km.cluster_centers_.astype(np.float32))
    selected = []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        if len(members) == 0:
            continue
        selected.append(members[int(np.argmin(raw[members] @ centers[cluster]))])
    return unique_fill(selected, len(raw), count, rng)


def language_kmeans_medoid(raw: np.ndarray, count: int, rng: np.random.Generator, languages: np.ndarray, **_) -> np.ndarray:
    selected = []
    labels = sorted(set(languages))
    for label in labels:
        members = np.flatnonzero(languages == label)
        if len(members) == 0:
            continue
        lang_count = max(1, round(count * len(members) / len(raw)))
        lang_count = min(lang_count, len(members), count - len(selected))
        if lang_count <= 0:
            continue
        km = MiniBatchKMeans(
            n_clusters=lang_count,
            random_state=int(rng.integers(0, 2**31 - 1)),
            batch_size=1024,
            n_init=2,
            max_iter=80,
        )
        labels_local = km.fit_predict(raw[members])
        centers = l2_normalize(km.cluster_centers_.astype(np.float32))
        for cluster in range(lang_count):
            local = np.flatnonzero(labels_local == cluster)
            if len(local):
                global_members = members[local]
                selected.append(global_members[int(np.argmax(raw[global_members] @ centers[cluster]))])
    return unique_fill(selected, len(raw), count, rng)


def density_anchors(
    raw: np.ndarray,
    count: int,
    rng: np.random.Generator,
    mode: str,
    k: int = 20,
    **_,
) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="cosine")
    nn.fit(raw)
    dist, _ = nn.kneighbors(raw, return_distance=True)
    density_score = dist[:, 1:].mean(axis=1)
    if mode == "dense":
        order = np.argsort(density_score)
    elif mode == "sparse":
        order = np.argsort(-density_score)
    else:
        dense = np.argsort(density_score)[: count // 2]
        sparse = np.argsort(-density_score)[: count - len(dense)]
        order = np.concatenate([dense, sparse])
    return unique_fill(list(order), len(raw), count, rng)


def multi_scale(raw: np.ndarray, count: int, rng: np.random.Generator, **_) -> np.ndarray:
    coarse_count = max(16, count // 4)
    coarse = list(kmeans_medoid(raw, coarse_count, rng))
    remaining = count - len(coarse)
    fps = list(farthest_point(raw, remaining, rng, start="centroid"))
    return unique_fill(coarse + fps, len(raw), count, rng)


def hard_negative(
    raw: np.ndarray,
    count: int,
    rng: np.random.Generator,
    query_indices: np.ndarray,
    truth_top: np.ndarray,
    candidate_pool: int = 100,
    seed_count: int | None = None,
    **_,
) -> np.ndarray:
    seed_count = seed_count or max(32, count // 2)
    selected = list(kmeans_medoid(raw, seed_count, rng))
    sig = l2_normalize(raw @ raw[np.array(selected)].T)
    missed = []
    for row, q_idx in enumerate(query_indices):
        rel_scores = sig[q_idx] @ sig.T
        rel_scores[q_idx] = -np.inf
        pool = set(top_indices(rel_scores, min(candidate_pool, len(raw) - 1)))
        misses = [idx for idx in truth_top[row] if idx not in pool]
        missed.extend(misses)
        if len(misses):
            missed.append(int(q_idx))
    if missed:
        counts = {}
        for idx in missed:
            counts[int(idx)] = counts.get(int(idx), 0) + 1
        selected.extend([idx for idx, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)])
    return unique_fill(selected, len(raw), count, rng)


def make_signatures(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return raw @ raw[anchors].T


def transform_signatures(sig: np.ndarray, name: str) -> np.ndarray:
    x = sig.astype(np.float32).copy()
    if name == "raw":
        return l2_normalize(x)
    if name == "centered":
        return l2_normalize(x - x.mean(axis=0, keepdims=True))
    if name == "zscore":
        return l2_normalize((x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-6))
    if name == "row_zscore":
        return l2_normalize((x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6))
    if name.startswith("softmax"):
        temp = float(name.split("_")[1])
        z = x / temp
        z -= z.max(axis=1, keepdims=True)
        return l2_normalize(np.exp(z))
    if name.startswith("top"):
        keep = int(name[3:])
        order = np.argpartition(-x, kth=min(keep - 1, x.shape[1] - 1), axis=1)[:, :keep]
        out = np.zeros_like(x)
        rows = np.arange(x.shape[0])[:, None]
        out[rows, order] = x[rows, order]
        return l2_normalize(out)
    if name.startswith("binary_top"):
        keep = int(name.replace("binary_top", ""))
        order = np.argpartition(-x, kth=min(keep - 1, x.shape[1] - 1), axis=1)[:, :keep]
        out = np.zeros_like(x)
        out[np.arange(x.shape[0])[:, None], order] = 1.0
        return l2_normalize(out)
    if name == "rank":
        ranks = np.empty_like(x, dtype=np.float32)
        order = np.argsort(-x, axis=1)
        ranks[np.arange(x.shape[0])[:, None], order] = np.arange(x.shape[1], dtype=np.float32)
        return l2_normalize(1.0 - ranks / max(1, x.shape[1] - 1))
    if name == "sign_centered":
        return l2_normalize((x > x.mean(axis=0, keepdims=True)).astype(np.float32))
    raise ValueError(f"Unknown transform: {name}")


def raw_truth(raw: np.ndarray, query_indices: np.ndarray, top_k: int, batch_size: int) -> np.ndarray:
    truth = []
    for start in range(0, len(query_indices), batch_size):
        batch = query_indices[start : start + batch_size]
        scores = raw[batch] @ raw.T
        for row, idx in enumerate(batch):
            scores[row, idx] = -np.inf
            truth.append(top_indices(scores[row], top_k))
    return np.vstack(truth)


def evaluate(rel: np.ndarray, query_indices: np.ndarray, truth: np.ndarray, pools: list[int], batch_size: int) -> list[dict[str, float]]:
    max_pool = max(pools)
    buckets = {pool: {"recall": [], "all": [], "any": []} for pool in pools}
    for start in range(0, len(query_indices), batch_size):
        batch = query_indices[start : start + batch_size]
        scores = rel[batch] @ rel.T
        for row, idx in enumerate(batch):
            scores[row, idx] = -np.inf
            ranked = top_indices(scores[row], max_pool)
            true = set(truth[start + row])
            for pool in pools:
                hits = len(true & set(ranked[:pool]))
                buckets[pool]["recall"].append(hits / len(true))
                buckets[pool]["all"].append(hits == len(true))
                buckets[pool]["any"].append(hits > 0)
    return [
        {
            "pool_size": pool,
            "mean_recall": float(np.mean(vals["recall"])),
            "all_top_k_contained": float(np.mean(vals["all"])),
            "any_hit": float(np.mean(vals["any"])),
        }
        for pool, vals in buckets.items()
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_best(rows: list[dict[str, object]], path: Path, pool: int) -> None:
    selected = [r for r in rows if r["pool_size"] == pool]
    selected.sort(key=lambda r: r["mean_recall"], reverse=True)
    selected = selected[:20]
    labels = [f"{r['anchor_strategy']} + {r['transform']}" for r in selected]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(selected)), [r["mean_recall"] for r in selected], color="#4C78A8")
    ax.set_yticks(range(len(selected)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Mean recall of raw top-k at pool {pool}")
    ax.set_title("Best anchor/signature ablations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, default=Path("experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy"))
    parser.add_argument("--corpus-path", type=Path, default=Path("experiments/tenk_minilm_candidate/data/processed/hf_google_code_x_glue_ct_code_to_text_n10000_seed7.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/tenk_minilm_candidate/anchor_ablation"))
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-queries", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="*", type=int, default=[100, 250, 500])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--strategies", nargs="*", default=[
        "random",
        "language_random",
        "farthest_random",
        "farthest_centroid",
        "pca_extremes",
        "kmeans_medoid",
        "kmeans_boundary",
        "language_kmeans_medoid",
        "density_dense",
        "density_sparse",
        "density_mixed",
        "multi_scale",
        "hard_negative",
    ])
    parser.add_argument("--transforms", nargs="*", default=[
        "raw",
        "centered",
        "zscore",
        "row_zscore",
        "rank",
        "top32",
        "top64",
        "binary_top32",
        "softmax_0.05",
        "softmax_0.1",
        "sign_centered",
    ])
    args = parser.parse_args()

    raw = l2_normalize(np.load(args.embedding_path).astype(np.float32))
    languages = load_languages(args.corpus_path, len(raw))
    rng = np.random.default_rng(args.seed)
    query_indices = np.sort(rng.choice(len(raw), size=min(args.sample_queries, len(raw)), replace=False))
    truth = raw_truth(raw, query_indices, args.top_k, args.batch_size)

    strategy_fns = {
        "random": lambda: random_anchors(raw, args.anchor_count, rng),
        "language_random": lambda: language_stratified_random(raw, args.anchor_count, rng, languages),
        "farthest_random": lambda: farthest_point(raw, args.anchor_count, rng, "random"),
        "farthest_centroid": lambda: farthest_point(raw, args.anchor_count, rng, "centroid"),
        "pca_extremes": lambda: pca_extremes(raw, args.anchor_count, rng),
        "kmeans_medoid": lambda: kmeans_medoid(raw, args.anchor_count, rng),
        "kmeans_boundary": lambda: kmeans_boundary(raw, args.anchor_count, rng),
        "language_kmeans_medoid": lambda: language_kmeans_medoid(raw, args.anchor_count, rng, languages),
        "density_dense": lambda: density_anchors(raw, args.anchor_count, rng, "dense"),
        "density_sparse": lambda: density_anchors(raw, args.anchor_count, rng, "sparse"),
        "density_mixed": lambda: density_anchors(raw, args.anchor_count, rng, "mixed"),
        "multi_scale": lambda: multi_scale(raw, args.anchor_count, rng),
        "hard_negative": lambda: hard_negative(raw, args.anchor_count, rng, query_indices, truth),
    }

    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for strategy in args.strategies:
        start = time.perf_counter()
        anchors = strategy_fns[strategy]()
        anchor_seconds = time.perf_counter() - start
        np.save(args.out_dir / f"anchors_{strategy}.npy", anchors)
        base_sig = make_signatures(raw, anchors)
        for transform in args.transforms:
            start = time.perf_counter()
            rel = transform_signatures(base_sig, transform)
            transform_seconds = time.perf_counter() - start
            start = time.perf_counter()
            metrics = evaluate(rel, query_indices, truth, args.pools, args.batch_size)
            eval_seconds = time.perf_counter() - start
            for metric in metrics:
                rows.append(
                    {
                        "anchor_strategy": strategy,
                        "transform": transform,
                        "top_k": args.top_k,
                        "query_count": len(query_indices),
                        "anchor_count": len(anchors),
                        "anchor_seconds": anchor_seconds,
                        "transform_seconds": transform_seconds,
                        "eval_seconds": eval_seconds,
                        **metric,
                    }
                )
            print(f"{strategy:24s} {transform:14s} pool{max(args.pools)} recall={metrics[-1]['mean_recall']:.4f}")

    write_csv(args.out_dir / "anchor_ablation.csv", rows)
    for pool in args.pools:
        plot_best(rows, args.out_dir / f"anchor_ablation_top20_pool{pool}.png", pool)

    best_pool = max(args.pools)
    best = sorted([r for r in rows if r["pool_size"] == best_pool], key=lambda r: r["mean_recall"], reverse=True)[:20]
    report = [
        "# Anchor Ablation",
        "",
        f"- Docs: {len(raw)}",
        f"- Anchors: {args.anchor_count}",
        f"- Queries: {len(query_indices)}",
        f"- Raw target: top-{args.top_k}",
        "",
        f"## Top Results At Pool {best_pool}",
        "",
        "| Anchor Strategy | Transform | Mean Recall | All Top-k | Any Hit |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best:
        report.append(
            f"| {row['anchor_strategy']} | {row['transform']} | {row['mean_recall']:.4f} | "
            f"{row['all_top_k_contained']:.4f} | {row['any_hit']:.4f} |"
        )
    (args.out_dir / "anchor_ablation.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
