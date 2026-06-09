import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_hnsw_index import LARPHNSWIndex, MultiGenerationLARP
from larp_index import l2_normalize, top_indices


def raw_truth(raw: np.ndarray, queries: np.ndarray, top_k: int) -> np.ndarray:
    out = []
    for q in queries:
        scores = raw @ raw[q]
        scores[q] = -np.inf
        out.append(top_indices(scores, top_k))
    return np.vstack(out)


def recall_rows(truth: np.ndarray, results: list[list[int]], top_k: int) -> tuple[float, float, float]:
    recalls = []
    all_hits = []
    any_hits = []
    for true, pred in zip(truth, results):
        hits = len(set(int(x) for x in true[:top_k]) & set(int(x) for x in pred))
        recalls.append(hits / top_k)
        all_hits.append(hits == top_k)
        any_hits.append(hits > 0)
    return float(np.mean(recalls)), float(np.mean(all_hits)), float(np.mean(any_hits))


def exact_raw_results(raw: np.ndarray, queries: np.ndarray, top_k: int) -> list[list[int]]:
    return [list(map(int, row)) for row in raw_truth(raw, queries, top_k)]


def raw_hnsw_results(index: LARPHNSWIndex, raw: np.ndarray, queries: np.ndarray, top_k: int) -> tuple[list[list[int]], float]:
    start = time.perf_counter()
    results = []
    for q in queries:
        labels, _ = index.raw_hnsw.knn_query(raw[q].reshape(1, -1), k=min(top_k + 1, index.size))
        results.append([int(idx) for idx in labels[0] if int(idx) != int(q)][:top_k])
    return results, 1000 * (time.perf_counter() - start) / max(1, len(queries))


def larp_results(index: LARPHNSWIndex, raw: np.ndarray, queries: np.ndarray, top_k: int, pool: int) -> tuple[list[list[int]], float]:
    start = time.perf_counter()
    results = []
    for q in queries:
        candidates, _ = index.candidate_indices(raw[q], pool=min(pool + 1, index.size))
        candidates = np.array([int(idx) for idx in candidates if int(idx) != int(q)], dtype=np.int64)
        raw_scores = raw[candidates] @ raw[q]
        order = top_indices(raw_scores, min(top_k, len(candidates)))
        results.append([int(candidates[pos]) for pos in order])
    return results, 1000 * (time.perf_counter() - start) / max(1, len(queries))


def relation_only_results(index: LARPHNSWIndex, raw: np.ndarray, queries: np.ndarray, top_k: int) -> tuple[list[list[int]], float]:
    start = time.perf_counter()
    results = []
    for q in queries:
        candidates, _ = index.candidate_indices(raw[q], pool=min(top_k + 1, index.size))
        results.append([int(idx) for idx in candidates if int(idx) != int(q)][:top_k])
    return results, 1000 * (time.perf_counter() - start) / max(1, len(queries))


def label_precision(labels: list[str], queries: np.ndarray, results: list[list[int]]) -> float:
    if not labels:
        return float("nan")
    vals = []
    for q, pred in zip(queries, results):
        q_label = labels[int(q)]
        vals.append(np.mean([labels[int(idx)] == q_label for idx in pred]) if pred else 0.0)
    return float(np.mean(vals))


def load_labels(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    labels = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            labels.append(str(row.get("language", "unknown")))
            if len(labels) >= n:
                break
    return labels


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_recall(rows: list[dict[str, object]], path: Path) -> None:
    selected = [r for r in rows if r["metric_group"] == "dynamic_insert" and int(r["top_k"]) == 10]
    methods = sorted({str(r["method"]) for r in selected})
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        subset = sorted([r for r in selected if r["method"] == method], key=lambda r: int(r["doc_count"]))
        ax.plot([int(r["doc_count"]) for r in subset], [float(r["mean_recall"]) for r in subset], marker="o", label=method)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Indexed documents after inserts")
    ax.set_ylabel("Recall of exact raw top-10")
    ax.set_title("Raw HNSW vs LARP-HNSW under incremental inserts")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def summarize_rows(rows: list[dict[str, object]], group: str, doc_count: int | None = None) -> list[list[object]]:
    selected = [r for r in rows if r["metric_group"] == group]
    if doc_count is not None:
        selected = [r for r in selected if int(r["doc_count"]) == doc_count]
    table = []
    for r in selected:
        table.append(
            [
                r["method"],
                r.get("pool", ""),
                f"{float(r['mean_recall']):.4f}",
                f"{float(r['all_top_k_contained']):.4f}",
                f"{float(r['ms_per_query']):.3f}",
                f"{float(r.get('label_precision_at_k', float('nan'))):.4f}",
            ]
        )
    return table


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    drift_rows: list[dict[str, object]],
    compression: dict[str, object],
    args: argparse.Namespace,
) -> None:
    final_n = max(int(r["doc_count"]) for r in rows if r["metric_group"] == "dynamic_insert")
    lines = [
        "# LARP HNSW System Benchmark",
        "",
        "## Scope",
        "",
        f"- Embedding file: `{args.embedding_path.as_posix()}`",
        f"- Corpus path: `{args.corpus_path.as_posix()}`",
        f"- Build docs: {args.build_docs}",
        f"- Final docs after inserts: {final_n}",
        f"- Query count per stage: {args.query_count}",
        f"- Top-k: {args.top_k}",
        f"- LARP candidate pools: {', '.join(str(p) for p in args.pools)}",
        "",
        "## What This Covers",
        "",
        "- Raw-HNSW baseline against exact raw cosine top-k.",
        "- LARP-HNSW over relative signatures with raw reranking.",
        "- Incremental inserts without recomputing existing signatures.",
        "- Drift diagnostics over new batches.",
        "- Multi-generation search simulation.",
        "- Relation-only top-k failure mode.",
        "- A weak label-based relevance proxy from corpus metadata.",
        "- Signature compression/memory estimates.",
        "",
        f"## Final Dynamic Insert Stage ({final_n} docs)",
        "",
        *markdown_table(["method", "pool", "recall", "all top-k", "ms/query", "label precision"], summarize_rows(rows, "dynamic_insert", final_n)),
        "",
        "## Relation-Only Check",
        "",
        *markdown_table(["method", "pool", "recall", "all top-k", "ms/query", "label precision"], summarize_rows(rows, "relation_only", final_n)),
        "",
        "## Multi-Generation Simulation",
        "",
        *markdown_table(["method", "pool", "recall", "all top-k", "ms/query", "label precision"], summarize_rows(rows, "generation", final_n)),
        "",
        "## Drift Diagnostics",
        "",
        *markdown_table(
            ["stage", "doc_count", "top-anchor sim", "entropy", "anchor gini"],
            [
                [
                    r["stage"],
                    r["doc_count"],
                    f"{float(r['mean_top_anchor_similarity']):.4f}",
                    f"{float(r['mean_signature_entropy']):.4f}",
                    f"{float(r['anchor_usage_gini']):.4f}",
                ]
                for r in drift_rows
            ],
        ),
        "",
        "## Compression Estimate",
        "",
        *markdown_table(
            ["representation", "MB"],
            [
                ["raw float32", f"{float(compression['raw_float32_mb']):.3f}"],
                ["signature float32", f"{float(compression['signature_float32_mb']):.3f}"],
                ["signature float16", f"{float(compression['signature_float16_mb']):.3f}"],
                ["signature int8", f"{float(compression['signature_int8_mb']):.3f}"],
            ],
        ),
        "",
        "## Interpretation",
        "",
        "The decisive comparison is raw-HNSW versus LARP-HNSW after inserts. LARP only has a practical advantage if it reaches comparable recall with lower latency, smaller routing memory, better insert behavior, or better degradation properties under append-heavy updates. Relation-only retrieval is reported separately because it is not the same claim as high-recall routing.",
        "",
        "The label precision column is a weak metadata proxy, not a human relevance benchmark. It is included only to avoid relying exclusively on raw embedding top-k as the evaluation target.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{(path.parent / 'larp_hnsw_system_benchmark.csv').as_posix()}`",
        f"- Drift CSV: `{(path.parent / 'larp_hnsw_drift.csv').as_posix()}`",
        f"- Plot: `{(path.parent / 'larp_hnsw_dynamic_recall.png').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_stage(
    index: LARPHNSWIndex,
    raw_all: np.ndarray,
    labels: list[str],
    indexed_n: int,
    queries: np.ndarray,
    args: argparse.Namespace,
    metric_group: str,
) -> list[dict[str, object]]:
    current_raw = raw_all[:indexed_n]
    truth = raw_truth(current_raw, queries, args.top_k)
    rows = []

    raw_results, raw_ms = raw_hnsw_results(index, current_raw, queries, args.top_k)
    rec, all_hit, any_hit = recall_rows(truth, raw_results, args.top_k)
    rows.append(
        {
            "metric_group": metric_group,
            "doc_count": indexed_n,
            "method": "raw_hnsw",
            "pool": args.top_k,
            "top_k": args.top_k,
            "query_count": len(queries),
            "mean_recall": rec,
            "all_top_k_contained": all_hit,
            "any_hit": any_hit,
            "ms_per_query": raw_ms,
            "label_precision_at_k": label_precision(labels, queries, raw_results),
        }
    )

    rel_results, rel_ms = relation_only_results(index, current_raw, queries, args.top_k)
    rec, all_hit, any_hit = recall_rows(truth, rel_results, args.top_k)
    rows.append(
        {
            "metric_group": "relation_only" if metric_group == "dynamic_insert" else metric_group,
            "doc_count": indexed_n,
            "method": "larp_hnsw_relation_only",
            "pool": args.top_k,
            "top_k": args.top_k,
            "query_count": len(queries),
            "mean_recall": rec,
            "all_top_k_contained": all_hit,
            "any_hit": any_hit,
            "ms_per_query": rel_ms,
            "label_precision_at_k": label_precision(labels, queries, rel_results),
        }
    )

    for pool in args.pools:
        results, ms = larp_results(index, current_raw, queries, args.top_k, pool)
        rec, all_hit, any_hit = recall_rows(truth, results, args.top_k)
        rows.append(
            {
                "metric_group": metric_group,
                "doc_count": indexed_n,
                "method": "larp_hnsw_rerank",
                "pool": pool,
                "top_k": args.top_k,
                "query_count": len(queries),
                "mean_recall": rec,
                "all_top_k_contained": all_hit,
                "any_hit": any_hit,
                "ms_per_query": ms,
                "label_precision_at_k": label_precision(labels, queries, results),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, default=Path("experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy"))
    parser.add_argument("--corpus-path", type=Path, default=Path("experiments/tenk_minilm_candidate/data/processed/hf_google_code_x_glue_ct_code_to_text_n10000_seed7.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/larp_hnsw_system"))
    parser.add_argument("--build-docs", type=int, default=8000)
    parser.add_argument("--insert-batches", nargs="+", type=int, default=[500, 500, 1000])
    parser.add_argument("--query-count", type=int, default=500)
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="+", type=int, default=[100, 250, 500])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ef-search", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = l2_normalize(np.load(args.embedding_path).astype(np.float32))
    final_n = min(len(raw), args.build_docs + sum(args.insert_batches))
    labels = load_labels(args.corpus_path, final_n)
    rng = np.random.default_rng(args.seed)

    index = LARPHNSWIndex(anchor_count=args.anchor_count, seed=args.seed, ef_search=args.ef_search, max_elements=final_n + 1)
    build_start = time.perf_counter()
    index.fit_embeddings(raw[: args.build_docs], doc_ids=[str(i) for i in range(args.build_docs)])
    build_seconds = time.perf_counter() - build_start

    rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    current_n = args.build_docs
    queries = np.sort(rng.choice(np.arange(current_n), size=min(args.query_count, current_n), replace=False))
    rows.extend(evaluate_stage(index, raw, labels, current_n, queries, args, "dynamic_insert"))
    drift = index.drift_stats(raw[:current_n])
    drift_rows.append({"stage": "build", **drift.__dict__, "build_seconds": build_seconds, "insert_ms_per_doc": 0.0})

    for batch_no, batch_size in enumerate(args.insert_batches, start=1):
        end = min(final_n, current_n + batch_size)
        if end <= current_n:
            break
        insert_start = time.perf_counter()
        index.insert_embeddings(raw[current_n:end], doc_ids=[str(i) for i in range(current_n, end)])
        insert_seconds = time.perf_counter() - insert_start
        current_n = end
        queries = np.sort(rng.choice(np.arange(current_n), size=min(args.query_count, current_n), replace=False))
        rows.extend(evaluate_stage(index, raw, labels, current_n, queries, args, "dynamic_insert"))
        drift = index.drift_stats(raw[current_n - batch_size : current_n])
        drift_rows.append(
            {
                "stage": f"insert_batch_{batch_no}",
                **drift.__dict__,
                "build_seconds": build_seconds,
                "insert_ms_per_doc": 1000 * insert_seconds / max(1, end - (current_n - batch_size)),
            }
        )

    # Generation protocol simulation: split old and new docs into separate anchor generations.
    gen_split = args.build_docs
    gen0 = LARPHNSWIndex(anchor_count=args.anchor_count, seed=args.seed, ef_search=args.ef_search)
    gen0.fit_embeddings(raw[:gen_split], doc_ids=[str(i) for i in range(gen_split)])
    gen1 = LARPHNSWIndex(anchor_count=min(args.anchor_count, final_n - gen_split), seed=args.seed + 1, ef_search=args.ef_search)
    gen1.fit_embeddings(raw[gen_split:final_n], doc_ids=[str(i) for i in range(gen_split, final_n)])
    multi = MultiGenerationLARP([gen0, gen1])
    queries = np.sort(rng.choice(np.arange(final_n), size=min(args.query_count, final_n), replace=False))
    truth = raw_truth(raw[:final_n], queries, args.top_k)
    start = time.perf_counter()
    generation_results = [
        [int(r.doc_id) for r in multi.search_embedding(raw[q], top_k=args.top_k + 1, pool_per_generation=max(args.pools)) if int(r.doc_id) != int(q)][: args.top_k]
        for q in queries
    ]
    generation_ms = 1000 * (time.perf_counter() - start) / max(1, len(queries))
    rec, all_hit, any_hit = recall_rows(truth, generation_results, args.top_k)
    rows.append(
        {
            "metric_group": "generation",
            "doc_count": final_n,
            "method": "multi_generation_larp",
            "pool": max(args.pools),
            "top_k": args.top_k,
            "query_count": len(queries),
            "mean_recall": rec,
            "all_top_k_contained": all_hit,
            "any_hit": any_hit,
            "ms_per_query": generation_ms,
            "label_precision_at_k": label_precision(labels, queries, generation_results),
        }
    )

    compression = index.memory_estimate().__dict__
    write_csv(args.out_dir / "larp_hnsw_system_benchmark.csv", rows)
    write_csv(args.out_dir / "larp_hnsw_drift.csv", drift_rows)
    plot_recall(rows, args.out_dir / "larp_hnsw_dynamic_recall.png")
    write_markdown(args.out_dir / "larp_hnsw_system_benchmark.md", rows, drift_rows, compression, args)
    index.save(args.out_dir / "saved_larp_hnsw_index")
    print(f"Wrote {args.out_dir / 'larp_hnsw_system_benchmark.md'}")


if __name__ == "__main__":
    main()
