import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from larp_index import LARPIndex, l2_normalize, top_indices


def raw_truth(raw: np.ndarray, query_indices: np.ndarray, top_k: int) -> list[set[int]]:
    truths = []
    for idx in query_indices:
        scores = raw @ raw[idx]
        scores[idx] = -np.inf
        truths.append(set(top_indices(scores, top_k)))
    return truths


def candidate_recall(index: LARPIndex, embeddings: np.ndarray, query_indices: np.ndarray, top_k: int, pool: int) -> dict[str, float]:
    raw = index.raw_embeddings
    truths = raw_truth(raw, query_indices, top_k)
    recalls = []
    all_hits = []
    search_start = time.perf_counter()
    for truth, query_idx in zip(truths, query_indices):
        query = embeddings[query_idx]
        query_sig = index._signature_from_raw(query.reshape(1, -1))[0]
        rel_scores = index.relative_signatures @ query_sig
        rel_scores[query_idx] = -np.inf
        candidates = set(top_indices(rel_scores, min(pool, len(rel_scores) - 1)))
        hits = len(truth & candidates)
        recalls.append(hits / top_k)
        all_hits.append(hits == top_k)
    elapsed = time.perf_counter() - search_start
    return {
        "query_count": len(query_indices),
        "top_k": top_k,
        "pool": pool,
        "mean_candidate_recall": float(np.mean(recalls)),
        "all_top_k_contained": float(np.mean(all_hits)),
        "candidate_ms_per_query": 1000.0 * elapsed / max(1, len(query_indices)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/tenk_minilm_candidate/larp_index_demo"))
    parser.add_argument("--base-docs", type=int, default=9900)
    parser.add_argument("--insert-docs", type=int, default=100)
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="*", type=int, default=[50, 100, 250, 500])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = l2_normalize(np.load(args.embedding_path).astype(np.float32))
    if args.base_docs + args.insert_docs > len(embeddings):
        raise ValueError("base-docs + insert-docs exceeds embedding count.")

    base_embeddings = embeddings[: args.base_docs]
    insert_embeddings = embeddings[args.base_docs : args.base_docs + args.insert_docs]
    base_ids = [f"doc-{i}" for i in range(args.base_docs)]

    build_start = time.perf_counter()
    index = LARPIndex(anchor_count=args.anchor_count, signature_transform="row_zscore", seed=args.seed)
    index.fit_embeddings(base_embeddings, base_ids)
    build_seconds = time.perf_counter() - build_start

    insert_start = time.perf_counter()
    insert_ids = [f"doc-{args.base_docs + offset}" for offset in range(len(insert_embeddings))]
    index.insert_embeddings(insert_embeddings, insert_ids)
    insert_seconds = time.perf_counter() - insert_start

    save_start = time.perf_counter()
    index.save(args.out_dir / "saved_index")
    save_seconds = time.perf_counter() - save_start

    load_start = time.perf_counter()
    loaded = LARPIndex.load(args.out_dir / "saved_index")
    load_seconds = time.perf_counter() - load_start

    rng = np.random.default_rng(args.seed)
    query_indices = np.sort(rng.choice(len(loaded.raw_embeddings), size=min(args.query_count, len(loaded.raw_embeddings)), replace=False))
    metric_rows = []
    for pool in args.pools:
        metric_rows.append(candidate_recall(loaded, loaded.raw_embeddings, query_indices, args.top_k, pool))

    rerank_bench = loaded.benchmark_queries(loaded.raw_embeddings[query_indices], top_k=args.top_k, pool=max(args.pools))

    summary = {
        "docs_after_insert": len(loaded.doc_ids),
        "base_docs": args.base_docs,
        "insert_docs": args.insert_docs,
        "anchor_count": args.anchor_count,
        "build_seconds": build_seconds,
        "insert_seconds_total": insert_seconds,
        "insert_ms_per_doc": 1000.0 * insert_seconds / max(1, args.insert_docs),
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "rerank_ms_per_query_pool_max": rerank_bench["ms_per_query"],
    }

    write_csv(args.out_dir / "larp_index_demo_metrics.csv", metric_rows)
    (args.out_dir / "larp_index_demo_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )

    report = [
        "# LARP Index Demo",
        "",
        f"- Base docs: {args.base_docs}",
        f"- Inserted docs: {args.insert_docs}",
        f"- Docs after insert: {len(loaded.doc_ids)}",
        f"- Anchors: {args.anchor_count}",
        f"- Build seconds: {build_seconds:.4f}",
        f"- Insert ms/doc: {summary['insert_ms_per_doc']:.4f}",
        f"- Save seconds: {save_seconds:.4f}",
        f"- Load seconds: {load_seconds:.4f}",
        f"- Full search + raw rerank at pool {max(args.pools)}: {rerank_bench['ms_per_query']:.4f} ms/query",
        "",
        "| Pool | Candidate Recall@Top-10 | All Top-10 Contained | Candidate ms/query |",
        "|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        report.append(
            f"| {row['pool']} | {row['mean_candidate_recall']:.4f} | "
            f"{row['all_top_k_contained']:.4f} | {row['candidate_ms_per_query']:.4f} |"
        )
    (args.out_dir / "larp_index_demo.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
