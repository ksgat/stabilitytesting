import argparse
import csv
import math
import time
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


def model_name(path: Path) -> str:
    name = path.stem
    for suffix in ("_n3000_tok160", "_n10000_tok160", "_n1000_tok160", "_n120_tok160"):
        name = name.replace(suffix, "")
    return name.replace("__", "/").replace("_", "-")


def farthest_point(raw: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    first = int(rng.integers(0, len(raw)))
    selected = [first]
    max_sim = raw @ raw[first]
    for _ in range(1, count):
        next_idx = int(np.argmin(max_sim))
        selected.append(next_idx)
        max_sim = np.maximum(max_sim, raw @ raw[next_idx])
    return np.array(selected, dtype=np.int32)


def random_anchors(raw: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(raw), size=count, replace=False).astype(np.int32)


def transform_signatures(sig: np.ndarray, transform: str) -> np.ndarray:
    x = sig.astype(np.float32, copy=True)
    if transform == "raw":
        return l2_normalize(x)
    if transform == "row_zscore":
        centered = x - x.mean(axis=1, keepdims=True)
        return l2_normalize(centered / np.maximum(x.std(axis=1, keepdims=True), 1e-6))
    raise ValueError(f"Unknown transform: {transform}")


def raw_truth(raw: np.ndarray, query_indices: np.ndarray, top_k: int, batch_size: int) -> np.ndarray:
    truth = []
    for start in range(0, len(query_indices), batch_size):
        batch = query_indices[start : start + batch_size]
        scores = raw[batch] @ raw.T
        for row, idx in enumerate(batch):
            scores[row, idx] = -np.inf
            truth.append(top_indices(scores[row], top_k))
    return np.vstack(truth)


def evaluate(
    rel: np.ndarray,
    query_indices: np.ndarray,
    truth: np.ndarray,
    pools: list[int],
    batch_size: int,
) -> tuple[list[dict[str, float]], float]:
    max_pool = max(pools)
    buckets = {pool: {"recall": [], "all": [], "any": []} for pool in pools}
    start_time = time.perf_counter()
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
    elapsed = time.perf_counter() - start_time
    rows = [
        {
            "pool_size": pool,
            "mean_recall": float(np.mean(vals["recall"])),
            "all_top_k_contained": float(np.mean(vals["all"])),
            "any_hit": float(np.mean(vals["any"])),
        }
        for pool, vals in buckets.items()
    ]
    return rows, elapsed * 1000.0 / max(1, len(query_indices))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_recall_curves(rows: list[dict[str, object]], path: Path, strategy: str) -> None:
    selected = [r for r in rows if r["strategy"] == strategy]
    models = sorted({str(r["model"]) for r in selected})
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        subset = sorted([r for r in selected if r["model"] == model], key=lambda r: int(r["pool_size"]))
        ax.plot(
            [int(r["pool_size"]) for r in subset],
            [float(r["mean_recall"]) for r in subset],
            marker="o",
            linewidth=2,
            label=model,
        )
    ax.set_xscale("log")
    ax.set_xticks(sorted({int(r["pool_size"]) for r in selected}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Candidate pool size")
    ax.set_ylabel("Raw top-10 recall in LARP candidate pool")
    ax.set_title("Best LARP configuration across embedding models")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_strategy_bars(rows: list[dict[str, object]], path: Path, pool: int) -> None:
    selected = [r for r in rows if int(r["pool_size"]) == pool]
    strategies = sorted({str(r["strategy"]) for r in selected})
    models = sorted({str(r["model"]) for r in selected})
    x = np.arange(len(models))
    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strategy in enumerate(strategies):
        vals = []
        for model in models:
            match = [r for r in selected if r["model"] == model and r["strategy"] == strategy]
            vals.append(float(match[0]["mean_recall"]) if match else math.nan)
        ax.bar(x + i * width, vals, width, label=strategy)
    ax.set_xticks(x + width * (len(strategies) - 1) / 2, models, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Raw top-10 recall")
    ax.set_title(f"Anchor/signature choices at pool {pool}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return out


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    embedding_files: list[Path],
    args: argparse.Namespace,
) -> None:
    best = "farthest_row_zscore"
    best_rows = [r for r in rows if r["strategy"] == best]
    pools = sorted({int(r["pool_size"]) for r in best_rows})
    model_rows = []
    for model in sorted({str(r["model"]) for r in best_rows}):
        subset = {int(r["pool_size"]): r for r in best_rows if r["model"] == model}
        model_rows.append(
            [
                model,
                subset[pools[0]]["embedding_dim"],
                subset[pools[0]]["doc_count"],
                *[f"{float(subset[p]['mean_recall']):.4f}" for p in pools],
                f"{float(subset[max(pools)]['search_ms_per_query']):.3f}",
            ]
        )

    pool_for_strategy = 250 if any(int(r["pool_size"]) == 250 for r in rows) else pools[-1]
    strategy_rows = []
    for strategy in sorted({str(r["strategy"]) for r in rows}):
        subset = [r for r in rows if r["strategy"] == strategy and int(r["pool_size"]) == pool_for_strategy]
        strategy_rows.append(
            [
                strategy,
                f"{np.mean([float(r['mean_recall']) for r in subset]):.4f}",
                f"{np.min([float(r['mean_recall']) for r in subset]):.4f}",
                f"{np.max([float(r['mean_recall']) for r in subset]):.4f}",
            ]
        )

    lines = [
        "# LARP Model Robustness Rerun",
        "",
        "## Scope",
        "",
        f"- Embedding files evaluated: {len(embedding_files)}",
        f"- Documents per embedding file: {args.n_docs or 'from file'}",
        f"- Sampled queries per model: {args.sample_queries}",
        f"- Anchor count: {args.anchor_count}",
        f"- Raw truth target: top-{args.top_k} nearest neighbors in each model's own embedding space",
        f"- Candidate pools: {', '.join(str(p) for p in args.pools)}",
        "",
        "## Method",
        "",
        "Each model is evaluated independently. Raw embeddings are L2-normalized, raw top-k neighbors are computed by cosine similarity, and LARP signatures are built by comparing every document to a fixed anchor set. The measured search step ranks documents by cosine similarity between relative signatures, then reports how many true raw top-k neighbors appear inside the relative candidate pool.",
        "",
        "Strategies tested:",
        "",
        "- `random_raw`: random anchors with L2-normalized raw anchor similarities.",
        "- `random_row_zscore`: random anchors with each document signature centered and scaled across anchors.",
        "- `farthest_raw`: farthest-point anchors with raw normalized signatures.",
        "- `farthest_row_zscore`: farthest-point anchors with row z-score signatures. This is the current best setting from the 10k anchor ablation.",
        "",
        "This is still a two-stage search test: the relative signature is evaluated as candidate generation, not as a final replacement for raw cosine reranking.",
        "",
        "## Best Configuration Results",
        "",
        *markdown_table(
            ["model", "dim", "docs", *[f"pool {p}" for p in pools], f"ms/query @ pool {max(pools)}"],
            model_rows,
        ),
        "",
        f"## Strategy Robustness at Pool {pool_for_strategy}",
        "",
        *markdown_table(["strategy", "mean recall", "min recall", "max recall"], strategy_rows),
        "",
        "## Findings",
        "",
        "- The robustness question is not whether signatures exactly replace embedding search; they do not yet. The useful behavior is whether a small, fixed relative vector reliably catches the raw neighbors as a candidate generator.",
        "- `farthest_row_zscore` is the main candidate for a stable growing index because anchor choice is fixed after build and new documents only need one embedding plus anchor comparisons.",
        "- If the best setting is consistently high across models at pool 250-500, the idea is probably not a single-model fluke. If one model collapses, the index needs model-specific calibration before it can be called universal.",
        "- The compute profile is favorable at this scale: signature search is matrix multiplication over `docs x anchors`, and insert cost is one row of raw embedding plus one row of anchor similarities.",
        "",
        "## Generated Artifacts",
        "",
        f"- CSV: `{(path.parent / 'model_robustness.csv').as_posix()}`",
        f"- Best-config plot: `{(path.parent / 'model_robustness_curves.png').as_posix()}`",
        f"- Strategy plot: `{(path.parent / f'model_robustness_strategy_pool{pool_for_strategy}.png').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-docs", type=int, default=0)
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--sample-queries", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.embedding_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy embedding files found in {args.embedding_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    strategies = [
        ("random_raw", random_anchors, "raw"),
        ("random_row_zscore", random_anchors, "row_zscore"),
        ("farthest_raw", farthest_point, "raw"),
        ("farthest_row_zscore", farthest_point, "row_zscore"),
    ]
    all_rows: list[dict[str, object]] = []

    for file in files:
        raw_loaded = np.load(file).astype(np.float32)
        if args.n_docs:
            raw_loaded = raw_loaded[: args.n_docs]
        raw = l2_normalize(raw_loaded)
        n_docs = len(raw)
        if args.anchor_count >= n_docs:
            raise ValueError(f"anchor-count {args.anchor_count} must be < doc count {n_docs}")
        rng = np.random.default_rng(args.seed)
        query_count = min(args.sample_queries, n_docs)
        query_indices = np.sort(rng.choice(n_docs, size=query_count, replace=False))
        truth = raw_truth(raw, query_indices, args.top_k, args.batch_size)

        for strategy_name, anchor_fn, transform in strategies:
            strategy_rng = np.random.default_rng(args.seed)
            anchor_start = time.perf_counter()
            anchors = anchor_fn(raw, args.anchor_count, strategy_rng)
            anchor_seconds = time.perf_counter() - anchor_start

            sig_start = time.perf_counter()
            sig = raw @ raw[anchors].T
            rel = transform_signatures(sig, transform)
            signature_seconds = time.perf_counter() - sig_start
            eval_rows, search_ms = evaluate(rel, query_indices, truth, args.pools, args.batch_size)
            for row in eval_rows:
                all_rows.append(
                    {
                        "model": model_name(file),
                        "embedding_file": str(file),
                        "embedding_dim": raw.shape[1],
                        "doc_count": n_docs,
                        "query_count": query_count,
                        "strategy": strategy_name,
                        "anchor_count": args.anchor_count,
                        "top_k": args.top_k,
                        "pool_size": row["pool_size"],
                        "mean_recall": row["mean_recall"],
                        "all_top_k_contained": row["all_top_k_contained"],
                        "any_hit": row["any_hit"],
                        "anchor_select_seconds": anchor_seconds,
                        "signature_build_seconds": signature_seconds,
                        "search_ms_per_query": search_ms,
                    }
                )

    csv_path = args.out_dir / "model_robustness.csv"
    write_csv(csv_path, all_rows)
    plot_recall_curves(all_rows, args.out_dir / "model_robustness_curves.png", "farthest_row_zscore")
    pool_for_strategy = 250 if 250 in args.pools else args.pools[-1]
    plot_strategy_bars(all_rows, args.out_dir / f"model_robustness_strategy_pool{pool_for_strategy}.png", pool_for_strategy)
    write_markdown(args.out_dir / "model_robustness.md", all_rows, files, args)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.out_dir / 'model_robustness.md'}")


if __name__ == "__main__":
    main()
