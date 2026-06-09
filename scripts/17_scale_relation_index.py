import argparse
import csv
import sys
import time
from pathlib import Path

import hnswlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_hnsw_index import select_farthest_anchors
from larp_index import l2_normalize, top_indices


def row_zscore(x: np.ndarray, batch_size: int = 20000) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size]
        centered = batch - batch.mean(axis=1, keepdims=True)
        scaled = centered / np.maximum(batch.std(axis=1, keepdims=True), 1e-6)
        out[start : start + batch_size] = l2_normalize(scaled.astype(np.float32))
    return out


def expand_embeddings(base: np.ndarray, n: int, noise: float, seed: int) -> np.ndarray:
    if n <= len(base):
        return base[:n].copy()
    rng = np.random.default_rng(seed)
    reps = rng.choice(len(base), size=n, replace=True)
    out = base[reps].copy()
    if noise > 0:
        out += rng.normal(0, noise, size=out.shape).astype(np.float32)
    return l2_normalize(out)


def relation_signatures(raw: np.ndarray, anchor_embeddings: np.ndarray, batch_size: int) -> np.ndarray:
    sig = np.empty((len(raw), len(anchor_embeddings)), dtype=np.float32)
    for start in range(0, len(raw), batch_size):
        sig[start : start + batch_size] = raw[start : start + batch_size] @ anchor_embeddings.T
    return row_zscore(sig, batch_size=batch_size)


def train_ridge(rel: np.ndarray, raw: np.ndarray, train_docs: np.ndarray, ridge: float) -> np.ndarray:
    s = rel[train_docs].astype(np.float64)
    y = raw[train_docs].astype(np.float64)
    lhs = s.T @ s
    lhs += ridge * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = s.T @ y
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def project_vectors(rel: np.ndarray, projection: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.empty((len(rel), projection.shape[1]), dtype=np.float32)
    for start in range(0, len(rel), batch_size):
        out[start : start + batch_size] = rel[start : start + batch_size] @ projection
    return l2_normalize(out)


def batched_topk(vectors: np.ndarray, queries: np.ndarray, top_k: int, score_batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(queries), score_batch_size):
        batch_queries = queries[start : start + score_batch_size].astype(np.int64)
        scores = vectors[batch_queries] @ vectors.T
        scores[np.arange(len(batch_queries)), batch_queries] = -np.inf
        if top_k >= scores.shape[1]:
            ranked = np.argsort(-scores, axis=1)
        else:
            part = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
            part_scores = np.take_along_axis(scores, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            ranked = np.take_along_axis(part, order, axis=1)
        rows.append(ranked.astype(np.int32))
    return np.vstack(rows)


def raw_truth(raw: np.ndarray, queries: np.ndarray, top_k: int, score_batch_size: int) -> np.ndarray:
    return batched_topk(raw, queries, top_k, score_batch_size)


def exact_recall(
    vectors: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    pools: list[int],
    score_batch_size: int,
) -> list[dict[str, float]]:
    max_pool = max(pools)
    buckets = {pool: {"recall": [], "all": []} for pool in pools}
    start_time = time.perf_counter()
    ranked_rows = batched_topk(vectors, queries, max_pool, score_batch_size)
    for row, ranked in enumerate(ranked_rows):
        true = set(int(x) for x in truth[row])
        for pool in pools:
            hits = len(true & set(int(x) for x in ranked[:pool]))
            buckets[pool]["recall"].append(hits / len(true))
            buckets[pool]["all"].append(hits == len(true))
    ms = 1000 * (time.perf_counter() - start_time) / max(1, len(queries))
    return [
        {
            "pool_size": pool,
            "mean_recall": float(np.mean(vals["recall"])),
            "all_top_k_contained": float(np.mean(vals["all"])),
            "ms_per_query": ms,
        }
        for pool, vals in buckets.items()
    ]


def build_hnsw(vectors: np.ndarray, ef_search: int, m: int, ef_construction: int) -> tuple[hnswlib.Index, float]:
    start = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), ef_construction=ef_construction, M=m)
    index.add_items(vectors, np.arange(len(vectors), dtype=np.int64))
    index.set_ef(ef_search)
    return index, time.perf_counter() - start


def hnsw_latency(index: hnswlib.Index, vectors: np.ndarray, queries: np.ndarray, k: int) -> float:
    start = time.perf_counter()
    index.knn_query(vectors[queries], k=k)
    return 1000 * (time.perf_counter() - start) / max(1, len(queries))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "metric",
        "doc_count",
        "data_mode",
        "method",
        "anchor_count",
        "ridge_reg",
        "pool_size",
        "mean_recall",
        "all_top_k_contained",
        "any_hit",
        "ms_per_query",
        "prep_seconds",
        "build_seconds",
        "vector_mb",
    ]
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(preferred))
    fieldnames = preferred + extras
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    selected = [r for r in rows if r["metric"] == "hnsw_latency"]
    methods = sorted({r["method"] for r in selected})
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        subset = sorted([r for r in selected if r["method"] == method], key=lambda r: int(r["doc_count"]))
        ax.plot([int(r["doc_count"]) for r in subset], [float(r["ms_per_query"]) for r in subset], marker="o", label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Documents / rows")
    ax.set_ylabel("HNSW ms/query")
    ax.set_title("Relation index scale mechanics")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def write_report(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    quality = [r for r in rows if r["metric"] == "quality" and int(r["pool_size"]) == 10]
    mechanics = [r for r in rows if r["metric"] == "hnsw_latency"]
    lines = [
        "# Relation Index Scale Benchmark",
        "",
        "## Scope",
        "",
        f"- Base embedding file: `{args.embedding_path.as_posix()}`",
        f"- Sizes: {', '.join(str(s) for s in args.sizes)}",
        f"- Anchors: {args.anchor_count}",
        f"- Ridge: {args.ridge_reg}",
        f"- Sizes above the base embedding count use noisy resampling and are mechanics-only. All listed sizes are real subsets when they are <= the base embedding count.",
        "",
        "## Real-Subset Relation-Only Quality at Pool 10",
        "",
        *markdown_table(
            ["docs", "method", "recall", "all top-k", "ms/query"],
            [
                [
                    r["doc_count"],
                    r["method"],
                    f"{float(r['mean_recall']):.4f}",
                    f"{float(r['all_top_k_contained']):.4f}",
                    f"{float(r['ms_per_query']):.3f}",
                ]
                for r in quality
            ],
        ),
        "",
        "## HNSW Mechanics",
        "",
        *markdown_table(
            ["docs", "mode", "method", "build s", "ms/query", "vector MB"],
            [
                [
                    r["doc_count"],
                    r["data_mode"],
                    r["method"],
                    f"{float(r['build_seconds']):.2f}",
                    f"{float(r['ms_per_query']):.3f}",
                    f"{float(r['vector_mb']):.1f}",
                ]
                for r in mechanics
            ],
        ),
        "",
        "## Interpretation",
        "",
        "Use real-subset rows for quality. Use synthetic rows only to estimate mechanics: vector memory, HNSW build time, and query latency. When 100k/200k are <= the base embedding count, they are real quality rows.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{(path.parent / 'relation_index_scale.csv').as_posix()}`",
        f"- Plot: `{(path.parent / 'relation_index_scale_latency.png').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, default=Path("experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/relation_index_scale"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 3000, 10000, 100000, 200000])
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--anchor-candidate-docs", type=int, default=20000)
    parser.add_argument("--ridge-reg", type=float, default=0.03)
    parser.add_argument("--eval-queries", type=int, default=500)
    parser.add_argument("--mechanics-queries", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="+", type=int, default=[10, 25])
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--skip-hnsw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = l2_normalize(np.load(args.embedding_path).astype(np.float32))
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for n in args.sizes:
        start_total = time.perf_counter()
        raw = expand_embeddings(base, n, args.noise, args.seed)
        data_mode = "real_subset" if n <= len(base) else "synthetic_expanded"
        train_docs = np.arange(max(100, int(n * 0.8)), dtype=np.int32)
        anchor_candidates = train_docs
        if len(anchor_candidates) > args.anchor_candidate_docs:
            anchor_candidates = np.sort(rng.choice(anchor_candidates, size=args.anchor_candidate_docs, replace=False)).astype(np.int32)
        anchors = select_farthest_anchors(raw, min(args.anchor_count, len(anchor_candidates)), args.seed, candidates=anchor_candidates)
        rel_start = time.perf_counter()
        rel = relation_signatures(raw, raw[anchors], args.batch_size)
        projection = train_ridge(rel, raw, train_docs, args.ridge_reg)
        projected = project_vectors(rel, projection, args.batch_size)
        prep_seconds = time.perf_counter() - rel_start

        query_pool = np.arange(int(n * 0.8), n, dtype=np.int32)
        if len(query_pool) == 0:
            query_pool = np.arange(n, dtype=np.int32)
        quality_queries = np.sort(rng.choice(query_pool, size=min(args.eval_queries, len(query_pool)), replace=False))
        mechanics_queries = np.sort(rng.choice(np.arange(n), size=min(args.mechanics_queries, n), replace=False))

        if data_mode == "real_subset":
            truth = raw_truth(raw, quality_queries, args.top_k, args.score_batch_size)
            for method, vectors in [("cosine_relation", rel), ("ridge_bilinear", projected)]:
                eval_rows = exact_recall(vectors, quality_queries, truth, args.pools, args.score_batch_size)
                for row in eval_rows:
                    rows.append(
                        {
                            "metric": "quality",
                            "doc_count": n,
                            "data_mode": data_mode,
                            "method": method,
                            "anchor_count": len(anchors),
                            "ridge_reg": args.ridge_reg if method == "ridge_bilinear" else 0.0,
                            "prep_seconds": prep_seconds,
                            "build_seconds": 0.0,
                            "vector_mb": n * vectors.shape[1] * 4 / 1_000_000,
                            **row,
                        }
                    )

        if not args.skip_hnsw:
            for method, vectors in [("ridge_bilinear", projected)]:
                index, build_seconds = build_hnsw(vectors, args.ef_search, args.hnsw_m, args.ef_construction)
                ms = hnsw_latency(index, vectors, mechanics_queries, args.top_k)
                rows.append(
                    {
                        "metric": "hnsw_latency",
                        "doc_count": n,
                        "data_mode": data_mode,
                        "method": method,
                        "anchor_count": len(anchors),
                        "ridge_reg": args.ridge_reg,
                        "pool_size": args.top_k,
                        "mean_recall": np.nan,
                        "all_top_k_contained": np.nan,
                        "any_hit": np.nan,
                        "ms_per_query": ms,
                        "prep_seconds": prep_seconds,
                        "build_seconds": build_seconds,
                        "vector_mb": n * vectors.shape[1] * 4 / 1_000_000,
                    }
                )
        print(f"n={n} mode={data_mode} anchors={len(anchors)} total_s={time.perf_counter() - start_total:.1f}", flush=True)

    write_csv(args.out_dir / "relation_index_scale.csv", rows)
    plot(rows, args.out_dir / "relation_index_scale_latency.png")
    write_report(args.out_dir / "relation_index_scale.md", rows, args)
    print(f"Wrote {args.out_dir / 'relation_index_scale.md'}")


if __name__ == "__main__":
    main()
