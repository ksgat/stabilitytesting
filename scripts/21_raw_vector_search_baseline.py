import argparse
import csv
import sys
import time
from pathlib import Path

import hnswlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_hnsw_index import select_farthest_anchors
from larp_index import l2_normalize


def row_zscore(x: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size]
        centered = batch - batch.mean(axis=1, keepdims=True)
        scaled = centered / np.maximum(batch.std(axis=1, keepdims=True), 1e-6)
        out[start : start + batch_size] = l2_normalize(scaled.astype(np.float32))
    return out


def batched_topk(vectors: np.ndarray, queries: np.ndarray, top_k: int, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size].astype(np.int64)
        scores = vectors[batch_queries] @ vectors.T
        scores[np.arange(len(batch_queries)), batch_queries] = -np.inf
        part = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        rows.append(np.take_along_axis(part, order, axis=1).astype(np.int32))
    return np.vstack(rows)


def exact_latency(vectors: np.ndarray, queries: np.ndarray, top_k: int, batch_size: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    ranked = batched_topk(vectors, queries, top_k, batch_size)
    ms = 1000 * (time.perf_counter() - start) / max(1, len(queries))
    return ranked, ms


def build_hnsw(vectors: np.ndarray, ef_search: int, m: int, ef_construction: int) -> tuple[hnswlib.Index, float]:
    start = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), ef_construction=ef_construction, M=m)
    index.add_items(vectors, np.arange(len(vectors), dtype=np.int64))
    index.set_ef(ef_search)
    return index, time.perf_counter() - start


def hnsw_ranked(index: hnswlib.Index, vectors: np.ndarray, queries: np.ndarray, pool: int) -> tuple[list[np.ndarray], float]:
    start = time.perf_counter()
    labels, _ = index.knn_query(vectors[queries], k=min(pool + 1, len(vectors)))
    ms = 1000 * (time.perf_counter() - start) / max(1, len(queries))
    ranked = []
    for query_idx, row in zip(queries, labels):
        filtered = [int(x) for x in row if int(x) != int(query_idx)]
        ranked.append(np.array(filtered[:pool], dtype=np.int32))
    return ranked, ms


def recall_rows(ranked_rows: list[np.ndarray] | np.ndarray, truth: np.ndarray, pools: list[int]) -> list[dict[str, float]]:
    rows = []
    for pool in pools:
        recalls = []
        all_hits = []
        for ranked, target in zip(ranked_rows, truth):
            true = set(int(x) for x in target)
            hits = len(true & set(int(x) for x in ranked[:pool]))
            recalls.append(hits / len(true))
            all_hits.append(hits == len(true))
        rows.append(
            {
                "pool_size": pool,
                "mean_recall": float(np.mean(recalls)),
                "all_top_k_contained": float(np.mean(all_hits)),
            }
        )
    return rows


def relation_projected(raw: np.ndarray, anchor_count: int, anchor_candidate_docs: int, ridge: float, seed: int, batch_size: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    train_docs = np.arange(max(100, int(len(raw) * 0.8)), dtype=np.int32)
    candidates = train_docs
    if len(candidates) > anchor_candidate_docs:
        candidates = np.sort(rng.choice(candidates, size=anchor_candidate_docs, replace=False)).astype(np.int32)
    start = time.perf_counter()
    anchors = select_farthest_anchors(raw, min(anchor_count, len(candidates)), seed, candidates=candidates)
    rel = np.empty((len(raw), len(anchors)), dtype=np.float32)
    anchor_vectors = raw[anchors]
    for offset in range(0, len(raw), batch_size):
        rel[offset : offset + batch_size] = raw[offset : offset + batch_size] @ anchor_vectors.T
    rel = row_zscore(rel, batch_size)
    s = rel[train_docs].astype(np.float64)
    y = raw[train_docs].astype(np.float64)
    lhs = s.T @ s
    lhs += ridge * np.eye(lhs.shape[0], dtype=np.float64)
    projection = np.linalg.solve(lhs, s.T @ y).astype(np.float32)
    projected = np.empty((len(raw), projection.shape[1]), dtype=np.float32)
    for offset in range(0, len(raw), batch_size):
        projected[offset : offset + batch_size] = rel[offset : offset + batch_size] @ projection
    return l2_normalize(projected), time.perf_counter() - start


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_count",
        "method",
        "pool_size",
        "mean_recall",
        "all_top_k_contained",
        "build_seconds",
        "prep_seconds",
        "ms_per_query",
        "vector_dim",
        "vector_mb",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def write_report(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    pool10 = [r for r in rows if int(r["pool_size"]) == 10]
    pool25 = [r for r in rows if int(r["pool_size"]) == 25 and r["method"] != "raw_exact"]
    lines = [
        "# Raw Vector Search Baseline",
        "",
        "## Scope",
        "",
        f"- Embeddings: `{args.embedding_path.as_posix()}`",
        f"- Sizes: {', '.join(str(x) for x in args.sizes)}",
        f"- Queries per size: {args.eval_queries}",
        f"- HNSW: M={args.hnsw_m}, ef_construction={args.ef_construction}, ef_search={args.ef_search}",
        f"- Relation metric: {args.anchor_count} anchors, ridge={args.ridge_reg}",
        "",
        "Truth is exact cosine top-10 over raw embeddings. `raw_exact` is therefore recall 1.0 by definition and reports brute-force exact latency.",
        "",
        "## Pool-10 Comparison",
        "",
        *markdown_table(
            ["docs", "method", "recall", "all top-k", "build s", "prep s", "ms/query", "vector MB"],
            [
                [
                    r["doc_count"],
                    r["method"],
                    f"{float(r['mean_recall']):.4f}",
                    f"{float(r['all_top_k_contained']):.4f}",
                    f"{float(r['build_seconds']):.2f}",
                    f"{float(r['prep_seconds']):.2f}",
                    f"{float(r['ms_per_query']):.3f}",
                    f"{float(r['vector_mb']):.1f}",
                ]
                for r in pool10
            ],
        ),
        "",
        "## Pool-25 Candidate Comparison",
        "",
        *markdown_table(
            ["docs", "method", "recall of raw top-10", "all top-10 contained"],
            [
                [
                    r["doc_count"],
                    r["method"],
                    f"{float(r['mean_recall']):.4f}",
                    f"{float(r['all_top_k_contained']):.4f}",
                ]
                for r in pool25
            ],
        ),
        "",
        "## Interpretation",
        "",
        "Raw HNSW is the direct vector-search baseline. Ridge-bilinear relation HNSW must beat this on latency, memory, freshness, or quality to justify itself as a primary search backend.",
        "",
        "On this run, raw HNSW wins as the primary top-10 backend. Ridge-relation HNSW is competitive only as a small candidate-pool generator: at pool 25 it nearly matches raw HNSW containment, but at direct top-10 it loses too much exact-neighbor recall.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{(path.parent / 'raw_vector_search_baseline.csv').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, default=Path("experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/real_distinct_hf_code/raw_vector_baseline"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[100000, 200000])
    parser.add_argument("--eval-queries", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="+", type=int, default=[10, 25])
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--anchor-candidate-docs", type=int, default=20000)
    parser.add_argument("--ridge-reg", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = l2_normalize(np.load(args.embedding_path).astype(np.float32))
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for n in args.sizes:
        raw = base[:n].copy()
        query_pool = np.arange(int(n * 0.8), n, dtype=np.int32)
        queries = np.sort(rng.choice(query_pool, size=min(args.eval_queries, len(query_pool)), replace=False))
        truth, raw_exact_ms = exact_latency(raw, queries, args.top_k, args.score_batch_size)
        for row in recall_rows(truth, truth, args.pools):
            rows.append(
                {
                    "doc_count": n,
                    "method": "raw_exact",
                    "build_seconds": 0.0,
                    "prep_seconds": 0.0,
                    "ms_per_query": raw_exact_ms,
                    "vector_dim": raw.shape[1],
                    "vector_mb": n * raw.shape[1] * 4 / 1_000_000,
                    "notes": "brute-force exact raw cosine",
                    **row,
                }
            )

        raw_index, raw_build = build_hnsw(raw, args.ef_search, args.hnsw_m, args.ef_construction)
        raw_ranked, raw_hnsw_ms = hnsw_ranked(raw_index, raw, queries, max(args.pools))
        for row in recall_rows(raw_ranked, truth, args.pools):
            rows.append(
                {
                    "doc_count": n,
                    "method": "raw_hnsw",
                    "build_seconds": raw_build,
                    "prep_seconds": 0.0,
                    "ms_per_query": raw_hnsw_ms,
                    "vector_dim": raw.shape[1],
                    "vector_mb": n * raw.shape[1] * 4 / 1_000_000,
                    "notes": "HNSW over raw embeddings",
                    **row,
                }
            )

        projected, relation_prep = relation_projected(
            raw,
            args.anchor_count,
            args.anchor_candidate_docs,
            args.ridge_reg,
            args.seed,
            args.batch_size,
        )
        relation_index, relation_build = build_hnsw(projected, args.ef_search, args.hnsw_m, args.ef_construction)
        relation_ranked, relation_ms = hnsw_ranked(relation_index, projected, queries, max(args.pools))
        for row in recall_rows(relation_ranked, truth, args.pools):
            rows.append(
                {
                    "doc_count": n,
                    "method": "ridge_relation_hnsw",
                    "build_seconds": relation_build,
                    "prep_seconds": relation_prep,
                    "ms_per_query": relation_ms,
                    "vector_dim": projected.shape[1],
                    "vector_mb": n * projected.shape[1] * 4 / 1_000_000,
                    "notes": "HNSW over ridge-bilinear projected relation vectors",
                    **row,
                }
            )
        print(f"n={n} done", flush=True)

    write_csv(args.out_dir / "raw_vector_search_baseline.csv", rows)
    write_report(args.out_dir / "raw_vector_search_baseline.md", rows, args)
    print(f"Wrote {args.out_dir / 'raw_vector_search_baseline.md'}")


if __name__ == "__main__":
    main()
