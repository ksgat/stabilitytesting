import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, kth=k - 1)[:k]
    return part[np.argsort(-scores[part])]


def recall_rows(
    method: str,
    candidate_lists: list[np.ndarray],
    true_top: np.ndarray,
    pools: list[int],
    elapsed_s: float,
    extra: dict[str, float],
) -> list[dict[str, object]]:
    rows = []
    n_queries = len(candidate_lists)
    for pool in pools:
        recalls = []
        hit_all = []
        any_hit = []
        candidate_counts = []
        for candidates, truth in zip(candidate_lists, true_top):
            use = candidates[: min(pool, len(candidates))]
            candidate_counts.append(len(use))
            hits = len(set(use) & set(truth))
            recalls.append(hits / len(truth))
            hit_all.append(hits == len(truth))
            any_hit.append(hits > 0)
        rows.append(
            {
                "method": method,
                "pool_size": pool,
                "query_count": n_queries,
                "mean_recall": float(np.mean(recalls)),
                "all_top_k_contained": float(np.mean(hit_all)),
                "any_hit": float(np.mean(any_hit)),
                "avg_candidates_scored": float(np.mean(candidate_counts)),
                "total_seconds": elapsed_s,
                "ms_per_query": 1000.0 * elapsed_s / n_queries,
                **extra,
            }
        )
    return rows


@dataclass
class RPNode:
    projection: np.ndarray | None = None
    threshold: float = 0.0
    left: "RPNode | None" = None
    right: "RPNode | None" = None
    indices: np.ndarray | None = None


class RPTree:
    def __init__(self, data: np.ndarray, rng: np.random.Generator, leaf_size: int, max_depth: int):
        self.data = data
        self.rng = rng
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.root = self._build(np.arange(len(data), dtype=np.int32), 0)

    def _build(self, indices: np.ndarray, depth: int) -> RPNode:
        if len(indices) <= self.leaf_size or depth >= self.max_depth:
            return RPNode(indices=indices)

        projection = self.rng.normal(size=self.data.shape[1]).astype(np.float32)
        projection /= max(np.linalg.norm(projection), 1e-12)
        values = self.data[indices] @ projection
        threshold = float(np.median(values))
        left_indices = indices[values <= threshold]
        right_indices = indices[values > threshold]
        if len(left_indices) == 0 or len(right_indices) == 0:
            return RPNode(indices=indices)

        return RPNode(
            projection=projection,
            threshold=threshold,
            left=self._build(left_indices, depth + 1),
            right=self._build(right_indices, depth + 1),
        )

    def collect(self, query: np.ndarray, beam_leaves: int) -> np.ndarray:
        candidates = []
        queue: list[tuple[float, RPNode]] = [(0.0, self.root)]
        leaves = 0
        while queue and leaves < beam_leaves:
            queue.sort(key=lambda item: item[0])
            _, node = queue.pop(0)
            if node.indices is not None:
                candidates.append(node.indices)
                leaves += 1
                continue

            value = float(query @ node.projection)
            near, far = (node.left, node.right) if value <= node.threshold else (node.right, node.left)
            margin = abs(value - node.threshold)
            queue.append((0.0, near))
            queue.append((margin, far))

        if not candidates:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(candidates))


class RPForest:
    def __init__(
        self,
        data: np.ndarray,
        n_trees: int,
        leaf_size: int,
        max_depth: int,
        seed: int,
    ):
        self.data = data
        rng = np.random.default_rng(seed)
        self.trees = [
            RPTree(data, np.random.default_rng(int(rng.integers(0, 2**31 - 1))), leaf_size, max_depth)
            for _ in range(n_trees)
        ]

    def search(self, query: np.ndarray, pool: int, beam_leaves: int) -> tuple[np.ndarray, int]:
        candidates = [tree.collect(query, beam_leaves) for tree in self.trees]
        candidates = np.unique(np.concatenate([c for c in candidates if len(c)]))
        if len(candidates) == 0:
            return np.empty(0, dtype=np.int32), 0
        scores = self.data[candidates] @ query
        ranked = candidates[top_indices(scores, min(pool, len(candidates)))]
        return ranked, len(candidates)


def exact_relative_candidates(rel: np.ndarray, query_indices: np.ndarray, max_pool: int, batch_size: int):
    candidates = []
    start = time.perf_counter()
    for batch_start in range(0, len(query_indices), batch_size):
        batch_indices = query_indices[batch_start : batch_start + batch_size]
        scores = rel[batch_indices] @ rel.T
        for row_offset, query_idx in enumerate(batch_indices):
            scores[row_offset, query_idx] = -np.inf
            candidates.append(top_indices(scores[row_offset], max_pool))
    return candidates, time.perf_counter() - start


def raw_truth(raw: np.ndarray, query_indices: np.ndarray, top_k: int, batch_size: int) -> np.ndarray:
    truth = []
    for batch_start in range(0, len(query_indices), batch_size):
        batch_indices = query_indices[batch_start : batch_start + batch_size]
        scores = raw[batch_indices] @ raw.T
        for row_offset, query_idx in enumerate(batch_indices):
            scores[row_offset, query_idx] = -np.inf
            truth.append(top_indices(scores[row_offset], top_k))
    return np.vstack(truth)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        selected.sort(key=lambda row: row["pool_size"])
        ax.plot(
            [row["pool_size"] for row in selected],
            [row["mean_recall"] for row in selected],
            marker="o",
            label=f"{method} ({selected[0]['ms_per_query']:.2f} ms/q)",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Candidate pool")
    ax.set_ylabel("Recall of raw top-k")
    ax.set_title("Relative-signature index traversal vs exact relative search")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/tenk_minilm_candidate/tree_index"))
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-queries", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="*", type=int, default=[50, 100, 250, 500, 1000])
    parser.add_argument("--exact-batch-size", type=int, default=128)
    parser.add_argument("--trees", type=int, default=24)
    parser.add_argument("--leaf-size", type=int, default=96)
    parser.add_argument("--beam-leaves", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=32)
    args = parser.parse_args()

    raw = l2_normalize(np.load(args.embedding_path).astype(np.float32))
    n_docs = len(raw)
    rng = np.random.default_rng(args.seed)
    anchor_indices = rng.permutation(n_docs)[: args.anchor_count]
    rel = l2_normalize(raw @ raw[anchor_indices].T)
    query_indices = np.sort(rng.choice(n_docs, size=min(args.sample_queries, n_docs), replace=False))
    max_pool = max(args.pools)

    print(f"docs={n_docs} dims={raw.shape[1]} anchors={args.anchor_count} queries={len(query_indices)}")
    truth = raw_truth(raw, query_indices, args.top_k, args.exact_batch_size)

    exact_candidates, exact_s = exact_relative_candidates(rel, query_indices, max_pool, args.exact_batch_size)
    rows = recall_rows(
        "exact_relative",
        exact_candidates,
        truth,
        args.pools,
        exact_s,
        {"n_trees": 0, "leaf_size": 0, "beam_leaves": 0, "build_seconds": 0.0},
    )

    build_start = time.perf_counter()
    forest = RPForest(rel, args.trees, args.leaf_size, args.max_depth, args.seed)
    build_s = time.perf_counter() - build_start

    forest_candidates = []
    candidate_counts = []
    search_start = time.perf_counter()
    for query_idx in query_indices:
        ranked, scored_count = forest.search(rel[query_idx], max_pool, args.beam_leaves)
        forest_candidates.append(ranked)
        candidate_counts.append(scored_count)
    forest_s = time.perf_counter() - search_start
    forest_rows = recall_rows(
        "rp_forest",
        forest_candidates,
        truth,
        args.pools,
        forest_s,
        {
            "n_trees": args.trees,
            "leaf_size": args.leaf_size,
            "beam_leaves": args.beam_leaves,
            "build_seconds": build_s,
        },
    )
    for row in forest_rows:
        row["avg_candidates_scored"] = float(np.mean(candidate_counts))
    rows.extend(forest_rows)

    write_csv(args.out_dir / "signature_index_benchmark.csv", rows)
    plot(rows, args.out_dir / "signature_index_benchmark.png")

    report = [
        "# Signature Index Benchmark",
        "",
        f"- Docs: {n_docs}",
        f"- Relative dimensions / anchors: {args.anchor_count}",
        f"- Query sample: {len(query_indices)}",
        f"- Raw target: top-{args.top_k}",
        f"- RP forest: {args.trees} trees, leaf size {args.leaf_size}, beam leaves {args.beam_leaves}",
        f"- RP forest build seconds: {build_s:.4f}",
        "",
        "| Method | Pool | Mean Recall | All Top-k | Any Hit | ms/query | Avg Scored |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['method']} | {row['pool_size']} | {row['mean_recall']:.4f} | "
            f"{row['all_top_k_contained']:.4f} | {row['any_hit']:.4f} | "
            f"{row['ms_per_query']:.4f} | {row['avg_candidates_scored']:.1f} |"
        )
    (args.out_dir / "signature_index_benchmark.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
