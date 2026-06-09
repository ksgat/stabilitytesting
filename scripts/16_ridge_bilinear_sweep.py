import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_hnsw_index import select_farthest_anchors
from larp_index import l2_normalize, top_indices


def row_zscore(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    return l2_normalize(scaled.astype(np.float32))


def col_zscore(x: np.ndarray, train_docs: np.ndarray) -> np.ndarray:
    mean = x[train_docs].mean(axis=0, keepdims=True)
    std = np.maximum(x[train_docs].std(axis=0, keepdims=True), 1e-6)
    return l2_normalize(((x - mean) / std).astype(np.float32))


def transform_signatures(raw_sigs: np.ndarray, train_docs: np.ndarray, transform: str) -> np.ndarray:
    if transform == "row_zscore":
        return row_zscore(raw_sigs)
    if transform == "raw_l2":
        return l2_normalize(raw_sigs.astype(np.float32))
    if transform == "col_zscore":
        return col_zscore(raw_sigs, train_docs)
    if transform == "row_col_zscore":
        return col_zscore(row_zscore(raw_sigs), train_docs)
    raise ValueError(f"Unknown transform: {transform}")


def raw_truth(raw: np.ndarray, query_indices: np.ndarray, top_k: int, batch_size: int = 128) -> np.ndarray:
    rows = []
    for start in range(0, len(query_indices), batch_size):
        batch = query_indices[start : start + batch_size]
        scores = raw[batch] @ raw.T
        for row, idx in enumerate(batch):
            scores[row, int(idx)] = -np.inf
            rows.append(top_indices(scores[row], top_k))
    return np.vstack(rows)


def evaluate(vectors: np.ndarray, query_indices: np.ndarray, truth: np.ndarray, pools: list[int], batch_size: int = 128) -> list[dict[str, float]]:
    max_pool = max(pools)
    buckets = {pool: {"recall": [], "all": [], "any": []} for pool in pools}
    start_time = time.perf_counter()
    for start in range(0, len(query_indices), batch_size):
        batch = query_indices[start : start + batch_size]
        scores = vectors[batch] @ vectors.T
        for row, idx in enumerate(batch):
            scores[row, int(idx)] = -np.inf
            ranked = top_indices(scores[row], max_pool)
            true = set(int(x) for x in truth[start + row])
            for pool in pools:
                hits = len(true & set(int(x) for x in ranked[:pool]))
                buckets[pool]["recall"].append(hits / len(true))
                buckets[pool]["all"].append(hits == len(true))
                buckets[pool]["any"].append(hits > 0)
    ms = 1000 * (time.perf_counter() - start_time) / max(1, len(query_indices))
    return [
        {
            "pool_size": pool,
            "mean_recall": float(np.mean(vals["recall"])),
            "all_top_k_contained": float(np.mean(vals["all"])),
            "any_hit": float(np.mean(vals["any"])),
            "ms_per_query": ms,
        }
        for pool, vals in buckets.items()
    ]


def train_ridge(rel: np.ndarray, raw_target: np.ndarray, train_docs: np.ndarray, ridge_reg: float, target: str) -> np.ndarray:
    s = rel[train_docs].astype(np.float64)
    if target == "raw":
        y = raw_target[train_docs].astype(np.float64)
    elif target == "pca_raw":
        centered = raw_target[train_docs].astype(np.float64)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        y = centered @ vt[: min(rel.shape[1], raw_target.shape[1])].T
    else:
        raise ValueError(f"Unknown target: {target}")
    lhs = s.T @ s
    lhs += ridge_reg * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = s.T @ y
    return np.linalg.solve(lhs, rhs).astype(np.float32)


FIELDNAMES = [
    "method",
    "anchor_count",
    "ridge_reg",
    "transform",
    "target",
    "doc_count",
    "train_docs",
    "eval_queries",
    "top_k",
    "fit_seconds",
    "pool_size",
    "mean_recall",
    "all_top_k_contained",
    "any_hit",
    "ms_per_query",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_best(rows: list[dict[str, object]], path: Path, pool: int) -> None:
    selected = [r for r in rows if int(r["pool_size"]) == pool]
    selected.sort(key=lambda r: float(r["mean_recall"]), reverse=True)
    selected = selected[:20]
    labels = [
        f"a{r['anchor_count']} reg{r['ridge_reg']} {r['transform']} {r['target']}"
        for r in selected
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(selected)), [float(r["mean_recall"]) for r in selected], color="#4C78A8")
    ax.set_yticks(range(len(selected)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(f"Relation-only recall at pool {pool}")
    ax.set_title("Best ridge-bilinear configurations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def write_report(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    selected10 = sorted([r for r in rows if int(r["pool_size"]) == 10], key=lambda r: float(r["mean_recall"]), reverse=True)
    selected25 = sorted([r for r in rows if int(r["pool_size"]) == 25], key=lambda r: float(r["mean_recall"]), reverse=True)
    best10 = selected10[:10]
    best25 = selected25[:10]

    def row_line(r: dict[str, object]) -> list[object]:
        return [
            r["method"],
            r["anchor_count"],
            r["ridge_reg"],
            r["transform"],
            r["target"],
            f"{float(r['mean_recall']):.4f}",
            f"{float(r['all_top_k_contained']):.4f}",
        ]

    lines = [
        "# Ridge Bilinear Sweep",
        "",
        "## Scope",
        "",
        f"- Embedding file: `{args.embedding_path.as_posix()}`",
        f"- Documents: {args.n_docs}",
        f"- Train docs: {args.train_docs}",
        f"- Eval queries: {args.eval_queries}",
        f"- Time budget minutes: {args.time_budget_minutes}",
        "",
        "## Best Pool-10 Configs",
        "",
        *markdown_table(["method", "anchors", "ridge", "transform", "target", "recall", "all top-k"], [row_line(r) for r in best10]),
        "",
        "## Best Pool-25 Configs",
        "",
        *markdown_table(["method", "anchors", "ridge", "transform", "target", "recall", "all top-k"], [row_line(r) for r in best25]),
        "",
        "## Finding",
        "",
        "This sweep is designed to find whether relation-only top-k improves from anchor count, signature normalization, and ridge regularization. The most important number is pool-10 recall because that is the closest test of final top-k without raw reranking.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{(path.parent / 'ridge_bilinear_sweep.csv').as_posix()}`",
        f"- Plot: `{(path.parent / 'ridge_bilinear_sweep_pool10.png').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, default=Path("experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/ridge_bilinear_sweep"))
    parser.add_argument("--n-docs", type=int, default=10000)
    parser.add_argument("--train-docs", type=int, default=8000)
    parser.add_argument("--eval-queries", type=int, default=1000)
    parser.add_argument("--anchor-counts", nargs="+", type=int, default=[256, 384, 512, 768])
    parser.add_argument("--ridge-regs", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--transforms", nargs="+", default=["row_zscore", "row_col_zscore", "col_zscore"])
    parser.add_argument("--targets", nargs="+", default=["raw"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="+", type=int, default=[10, 25, 50, 100, 250])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-budget-minutes", type=float, default=115.0)
    parser.add_argument("--save-projections", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "ridge_bilinear_sweep.csv"
    if csv_path.exists():
        csv_path.unlink()

    raw = l2_normalize(np.load(args.embedding_path).astype(np.float32)[: args.n_docs])
    train_docs = np.arange(args.train_docs, dtype=np.int32)
    eval_docs = np.arange(args.train_docs, len(raw), dtype=np.int32)
    rng = np.random.default_rng(args.seed)
    eval_queries = np.sort(rng.choice(eval_docs, size=min(args.eval_queries, len(eval_docs)), replace=False))
    truth = raw_truth(raw, eval_queries, args.top_k, args.batch_size)

    start_time = time.perf_counter()
    deadline = start_time + args.time_budget_minutes * 60
    rows: list[dict[str, object]] = []
    baseline_done: set[tuple[int, str]] = set()

    for anchor_count in args.anchor_counts:
        if time.perf_counter() > deadline:
            break
        anchors = select_farthest_anchors(raw, min(anchor_count, len(train_docs)), args.seed, candidates=train_docs)
        raw_sigs = raw @ raw[anchors].T
        for transform, target, ridge_reg in itertools.product(args.transforms, args.targets, args.ridge_regs):
            if time.perf_counter() > deadline:
                break
            rel = transform_signatures(raw_sigs, train_docs, transform)
            baseline_key = (len(anchors), transform)
            if baseline_key not in baseline_done:
                baseline_done.add(baseline_key)
                baseline = evaluate(rel, eval_queries, truth, args.pools, args.batch_size)
                for row in baseline:
                    out = {
                        "method": "cosine_relation",
                        "anchor_count": len(anchors),
                        "ridge_reg": 0.0,
                        "transform": transform,
                        "target": "none",
                        "doc_count": len(raw),
                        "train_docs": args.train_docs,
                        "eval_queries": len(eval_queries),
                        "top_k": args.top_k,
                        "fit_seconds": 0.0,
                        **row,
                    }
                    rows.append(out)
                    append_csv(csv_path, out)

            fit_start = time.perf_counter()
            projection = train_ridge(rel, raw, train_docs, ridge_reg, target)
            projected = l2_normalize(rel @ projection)
            fit_seconds = time.perf_counter() - fit_start
            eval_rows = evaluate(projected, eval_queries, truth, args.pools, args.batch_size)
            if args.save_projections:
                np.save(args.out_dir / f"ridge_projection_a{len(anchors)}_{transform}_{target}_reg{ridge_reg:g}.npy", projection)
            for row in eval_rows:
                out = {
                    "method": "ridge_bilinear",
                    "anchor_count": len(anchors),
                    "ridge_reg": ridge_reg,
                    "transform": transform,
                    "target": target,
                    "doc_count": len(raw),
                    "train_docs": args.train_docs,
                    "eval_queries": len(eval_queries),
                    "top_k": args.top_k,
                    "fit_seconds": fit_seconds,
                    **row,
                }
                rows.append(out)
                append_csv(csv_path, out)
            best10 = max([r for r in rows if int(r["pool_size"]) == 10], key=lambda r: float(r["mean_recall"]))
            print(
                f"anchors={len(anchors)} transform={transform} target={target} reg={ridge_reg:g} "
                f"best_pool10={best10['method']} {float(best10['mean_recall']):.4f}"
            )

    plot_best(rows, args.out_dir / "ridge_bilinear_sweep_pool10.png", 10)
    write_report(args.out_dir / "ridge_bilinear_sweep.md", rows, args)
    print(f"Wrote {args.out_dir / 'ridge_bilinear_sweep.md'}")


if __name__ == "__main__":
    main()
