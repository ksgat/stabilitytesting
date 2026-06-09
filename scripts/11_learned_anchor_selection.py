import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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


def unique_fill(selected: list[int], candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    seen = set()
    out = []
    for idx in selected:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        if len(out) >= count:
            return np.array(out, dtype=np.int32)
    for idx in rng.permutation(candidates):
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        if len(out) >= count:
            break
    return np.array(out, dtype=np.int32)


def farthest_point(raw: np.ndarray, candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    first = int(candidates[int(rng.integers(0, len(candidates)))])
    selected = [first]
    max_sim = raw[candidates] @ raw[first]
    for _ in range(1, count):
        next_idx = int(candidates[int(np.argmin(max_sim))])
        selected.append(next_idx)
        max_sim = np.maximum(max_sim, raw[candidates] @ raw[next_idx])
    return unique_fill(selected, candidates, count, rng)


def random_anchors(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(candidates, size=count, replace=False).astype(np.int32)


def row_zscore(sig: np.ndarray) -> np.ndarray:
    centered = sig - sig.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(sig.std(axis=1, keepdims=True), 1e-6)
    return l2_normalize(scaled.astype(np.float32))


def make_rel(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return row_zscore(raw @ raw[anchors].T)


def raw_truth(
    raw: np.ndarray,
    query_indices: np.ndarray,
    top_k: int,
    batch_size: int,
    search_indices: np.ndarray | None = None,
) -> np.ndarray:
    search = np.arange(len(raw), dtype=np.int32) if search_indices is None else search_indices.astype(np.int32)
    search_pos = {int(idx): pos for pos, idx in enumerate(search)}
    truth = []
    for start in range(0, len(query_indices), batch_size):
        batch = query_indices[start : start + batch_size]
        scores = raw[batch] @ raw[search].T
        for row, idx in enumerate(batch):
            pos = search_pos.get(int(idx))
            if pos is not None:
                scores[row, pos] = -np.inf
            truth.append(search[top_indices(scores[row], top_k)])
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


def hard_negatives(
    rel: np.ndarray,
    train_queries: np.ndarray,
    truth: np.ndarray,
    count: int,
    pool: int,
    rng: np.random.Generator,
    search_indices: np.ndarray | None = None,
) -> np.ndarray:
    search = np.arange(len(rel), dtype=np.int32) if search_indices is None else search_indices.astype(np.int32)
    search_pos = {int(idx): pos for pos, idx in enumerate(search)}
    out = []
    for row, q_idx in enumerate(train_queries):
        scores = rel[q_idx] @ rel[search].T
        pos = search_pos.get(int(q_idx))
        if pos is not None:
            scores[pos] = -np.inf
        ranked = search[top_indices(scores, min(pool, len(search) - 1))]
        blocked = set(int(x) for x in truth[row])
        blocked.add(int(q_idx))
        negs = [int(x) for x in ranked if int(x) not in blocked]
        if len(negs) < count:
            fill = [int(x) for x in rng.choice(search, size=count * 3, replace=True) if int(x) not in blocked]
            negs.extend(fill)
        out.append(negs[:count])
    return np.array(out, dtype=np.int32)


def train_anchor_weights(
    candidate_rel: np.ndarray,
    train_queries: np.ndarray,
    train_truth: np.ndarray,
    train_negs: np.ndarray,
    steps: int,
    batch_size: int,
    neg_count: int,
    lr: float,
    temp: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, list[float]]:
    torch.manual_seed(seed)
    sig = torch.from_numpy(candidate_rel).to(device)
    logits = torch.zeros(sig.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    train_rows = np.arange(len(train_queries))

    for _ in range(steps):
        batch_rows = rng.choice(train_rows, size=min(batch_size, len(train_rows)), replace=False)
        q_idx = torch.as_tensor(train_queries[batch_rows], dtype=torch.long, device=device)
        pos_np = np.array([rng.choice(train_truth[row]) for row in batch_rows], dtype=np.int64)
        neg_np = np.vstack(
            [rng.choice(train_negs[row], size=neg_count, replace=len(train_negs[row]) < neg_count) for row in batch_rows]
        )
        doc_np = np.concatenate([pos_np[:, None], neg_np], axis=1)
        doc_idx = torch.as_tensor(doc_np, dtype=torch.long, device=device)

        weights = F.softplus(logits) + 1e-6
        q = F.normalize(sig[q_idx] * weights, dim=1)
        docs = F.normalize(sig[doc_idx] * weights, dim=2)
        scores = torch.einsum("bd,bkd->bk", q, docs) / temp
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
        loss = F.cross_entropy(scores, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    weights = (F.softplus(logits) + 1e-6).detach().cpu().numpy().astype(np.float32)
    return weights, losses


def weighted_rel(candidate_rel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return l2_normalize(candidate_rel * weights[None, :])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pool(rows: list[dict[str, object]], path: Path, pool: int) -> None:
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
            vals.append(float(match[0]["mean_recall"]) if match else np.nan)
        ax.bar(x + i * width, vals, width, label=strategy)
    ax.set_xticks(x + width * (len(strategies) - 1) / 2, models, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Held-out raw top-10 recall")
    ax.set_title(f"Learned anchor selection vs baselines at pool {pool}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    pool = 250 if any(int(r["pool_size"]) == 250 for r in rows) else args.pools[-1]
    selected = [r for r in rows if int(r["pool_size"]) == pool]
    pool10 = 10 if any(int(r["pool_size"]) == 10 for r in rows) else min(args.pools)
    selected10 = [r for r in rows if int(r["pool_size"]) == pool10]
    model_rows = []
    for model in sorted({str(r["model"]) for r in rows}):
        subset = [r for r in selected if r["model"] == model]
        by_strategy = {str(r["strategy"]): r for r in subset}
        base = by_strategy.get("farthest_row_zscore")
        learned = by_strategy.get("learned_top256")
        weighted = by_strategy.get("learned_weighted_bank")
        delta = ""
        if base and learned:
            delta = f"{float(learned['mean_recall']) - float(base['mean_recall']):+.4f}"
        model_rows.append(
            [
                model,
                f"{float(base['mean_recall']):.4f}" if base else "",
                f"{float(learned['mean_recall']):.4f}" if learned else "",
                delta,
                f"{float(weighted['mean_recall']):.4f}" if weighted else "",
            ]
        )

    strategy_rows = []
    for strategy in sorted({str(r["strategy"]) for r in rows}):
        vals = [float(r["mean_recall"]) for r in selected if r["strategy"] == strategy]
        strategy_rows.append([strategy, f"{np.mean(vals):.4f}", f"{np.min(vals):.4f}", f"{np.max(vals):.4f}"])

    topk_rows = []
    for strategy in sorted({str(r["strategy"]) for r in rows}):
        vals = [float(r["mean_recall"]) for r in selected10 if r["strategy"] == strategy]
        topk_rows.append([strategy, f"{np.mean(vals):.4f}", f"{np.min(vals):.4f}", f"{np.max(vals):.4f}"])

    pool_summary_rows = []
    summary_pools = [p for p in [25, 50, 100, 250] if any(int(r["pool_size"]) == p for r in rows)]
    for strategy in sorted({str(r["strategy"]) for r in rows}):
        row = [strategy]
        for p in summary_pools:
            vals = [float(r["mean_recall"]) for r in rows if r["strategy"] == strategy and int(r["pool_size"]) == p]
            row.append(f"{np.mean(vals):.4f}")
        pool_summary_rows.append(row)

    lines = [
        "# Learned Anchor Selection",
        "",
        "## Scope",
        "",
        f"- Models evaluated: {len(set(str(r['model']) for r in rows))}",
        f"- Documents per model: {args.n_docs}",
        f"- Base/old docs used for anchor candidates: {args.train_docs}",
        f"- Held-out new-doc queries: {args.eval_queries}",
        f"- Candidate anchor bank: {args.candidate_anchor_count}",
        f"- Final selected anchors: {args.anchor_count}",
        f"- Training steps per model: {args.steps}",
        "",
        "## Method",
        "",
        "This tests whether anchor selection can be trained as a separate lightweight model. For each embedding model, the script builds a larger farthest-point candidate anchor bank from the first/base documents, computes relative signatures to that bank, then trains one positive weight per candidate anchor with a contrastive hard-negative loss using only base-document positives and negatives. After training, it keeps the highest-weighted 256 anchors and evaluates on held-out queries from later documents against the full corpus.",
        "",
        "Strategies:",
        "",
        "- `random_row_zscore`: random 256 base-document anchors.",
        "- `farthest_row_zscore`: 256 farthest-point base-document anchors.",
        "- `learned_top256`: top 256 anchors selected by the trained anchor weights.",
        "- `learned_weighted_bank`: all candidate anchors with learned weights, included as an upper-bound diagnostic rather than a fair storage match.",
        "",
        f"## Held-Out Results at Pool {pool}",
        "",
        *markdown_table(["model", "farthest", "learned top256", "delta", "weighted bank"], model_rows),
        "",
        "## Strategy Summary",
        "",
        *markdown_table(["strategy", "mean", "min", "max"], strategy_rows),
        "",
        f"## Relation-Only Top-{args.top_k} Check",
        "",
        f"Pool {pool10} is the closest proxy here for relation-only top-{args.top_k}, because the relative signature retrieves exactly {pool10} candidates while raw embedding top-{args.top_k} is treated as ground truth. Learned anchors can improve this case without making it a complete replacement for raw reranking.",
        "",
        *markdown_table([f"strategy", f"mean pool-{pool10} recall", "min", "max"], topk_rows),
        "",
        "Mean recall by pool:",
        "",
        *markdown_table(["strategy", *[f"pool {p}" for p in summary_pools]], pool_summary_rows),
        "",
        "## Finding",
        "",
        "`learned_top256` should be treated as a direct routing upgrade if it beats `farthest_row_zscore` on held-out queries. If the smallest-pool recall stays well below the larger-pool recall, relation-only top-k is still not strong enough to eliminate raw reranking.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{(path.parent / 'learned_anchor_selection.csv').as_posix()}`",
        f"- Plot: `{(path.parent / f'learned_anchor_selection_pool{pool}.png').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-docs", type=int, default=3000)
    parser.add_argument("--train-docs", type=int, default=2000)
    parser.add_argument("--train-queries", type=int, default=800)
    parser.add_argument("--eval-queries", type=int, default=800)
    parser.add_argument("--candidate-anchor-count", type=int, default=512)
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--neg-count", type=int, default=32)
    parser.add_argument("--hard-pool", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.embedding_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files found in {args.embedding_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    for file in files:
        name = model_name(file)
        raw = l2_normalize(np.load(file).astype(np.float32)[: args.n_docs])
        n_docs = len(raw)
        if args.train_docs >= n_docs:
            raise ValueError("--train-docs must be smaller than --n-docs")

        base_docs = np.arange(args.train_docs, dtype=np.int32)
        eval_pool = np.arange(args.train_docs, n_docs, dtype=np.int32)
        train_queries = np.sort(rng.choice(base_docs, size=min(args.train_queries, len(base_docs)), replace=False))
        eval_queries = np.sort(rng.choice(eval_pool, size=min(args.eval_queries, len(eval_pool)), replace=False))

        train_truth = raw_truth(raw, train_queries, args.top_k, args.batch_size, search_indices=base_docs)
        eval_truth = raw_truth(raw, eval_queries, args.top_k, args.batch_size)

        candidate_count = min(args.candidate_anchor_count, len(base_docs))
        candidate_anchors = farthest_point(raw, base_docs, candidate_count, np.random.default_rng(args.seed))
        farthest_anchors = candidate_anchors[: args.anchor_count]
        random_anchor_set = random_anchors(base_docs, args.anchor_count, np.random.default_rng(args.seed))

        strategies = {
            "random_row_zscore": make_rel(raw, random_anchor_set),
            "farthest_row_zscore": make_rel(raw, farthest_anchors),
        }

        baseline_rel = strategies["farthest_row_zscore"]
        train_negs = hard_negatives(
            baseline_rel,
            train_queries,
            train_truth,
            args.neg_count,
            args.hard_pool,
            np.random.default_rng(args.seed),
            search_indices=base_docs,
        )

        candidate_rel = make_rel(raw, candidate_anchors)
        train_start = time.perf_counter()
        weights, losses = train_anchor_weights(
            candidate_rel,
            train_queries,
            train_truth,
            train_negs,
            args.steps,
            args.batch_size,
            args.neg_count,
            args.lr,
            args.temp,
            args.seed,
            args.device,
        )
        train_seconds = time.perf_counter() - train_start

        selected = np.argsort(-weights)[: args.anchor_count]
        learned_anchors = candidate_anchors[selected]
        strategies["learned_top256"] = make_rel(raw, learned_anchors)
        strategies["learned_weighted_bank"] = weighted_rel(candidate_rel, weights)

        for strategy, rel in strategies.items():
            eval_rows, search_ms = evaluate(rel, eval_queries, eval_truth, args.pools, args.batch_size)
            for row in eval_rows:
                all_rows.append(
                    {
                        "model": name,
                        "embedding_file": str(file),
                        "strategy": strategy,
                        "doc_count": n_docs,
                        "train_docs": args.train_docs,
                        "train_queries": len(train_queries),
                        "eval_queries": len(eval_queries),
                        "candidate_anchor_count": candidate_count,
                        "anchor_count": args.anchor_count if strategy != "learned_weighted_bank" else candidate_count,
                        "top_k": args.top_k,
                        "pool_size": row["pool_size"],
                        "mean_recall": row["mean_recall"],
                        "all_top_k_contained": row["all_top_k_contained"],
                        "any_hit": row["any_hit"],
                        "search_ms_per_query": search_ms,
                        "train_seconds": train_seconds if strategy.startswith("learned") else 0.0,
                        "final_train_loss": losses[-1],
                    }
                )
        print(f"{name}: trained selector in {train_seconds:.1f}s; final loss {losses[-1]:.4f}")

    csv_path = args.out_dir / "learned_anchor_selection.csv"
    write_csv(csv_path, all_rows)
    plot_pool_value = 250 if 250 in args.pools else args.pools[-1]
    plot_pool(all_rows, args.out_dir / f"learned_anchor_selection_pool{plot_pool_value}.png", plot_pool_value)
    write_markdown(args.out_dir / "learned_anchor_selection.md", all_rows, args)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.out_dir / 'learned_anchor_selection.md'}")


if __name__ == "__main__":
    main()
