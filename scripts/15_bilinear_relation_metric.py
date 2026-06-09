import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_hnsw_index import select_farthest_anchors
from larp_index import l2_normalize, top_indices


def row_zscore(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    return l2_normalize(scaled.astype(np.float32))


def make_signatures(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return row_zscore(raw @ raw[anchors].T)


def raw_truth(
    raw: np.ndarray,
    query_indices: np.ndarray,
    top_k: int,
    search_indices: np.ndarray | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    search = np.arange(len(raw), dtype=np.int32) if search_indices is None else search_indices.astype(np.int32)
    search_pos = {int(idx): pos for pos, idx in enumerate(search)}
    rows = []
    for start in range(0, len(query_indices), batch_size):
        batch = query_indices[start : start + batch_size]
        scores = raw[batch] @ raw[search].T
        for row, idx in enumerate(batch):
            pos = search_pos.get(int(idx))
            if pos is not None:
                scores[row, pos] = -np.inf
            rows.append(search[top_indices(scores[row], top_k)])
    return np.vstack(rows)


def relation_hard_negatives(
    rel: np.ndarray,
    query_indices: np.ndarray,
    truth: np.ndarray,
    neg_count: int,
    pool: int,
    search_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    search = search_indices.astype(np.int32)
    search_pos = {int(idx): pos for pos, idx in enumerate(search)}
    rows = []
    for row, q_idx in enumerate(query_indices):
        scores = rel[int(q_idx)] @ rel[search].T
        pos = search_pos.get(int(q_idx))
        if pos is not None:
            scores[pos] = -np.inf
        ranked = search[top_indices(scores, min(pool, len(search) - 1))]
        blocked = set(int(x) for x in truth[row])
        blocked.add(int(q_idx))
        negs = [int(x) for x in ranked if int(x) not in blocked]
        if len(negs) < neg_count:
            fill = [int(x) for x in rng.choice(search, size=neg_count * 4, replace=True) if int(x) not in blocked]
            negs.extend(fill)
        rows.append(negs[:neg_count])
    return np.array(rows, dtype=np.int32)


def evaluate(
    vectors: np.ndarray,
    query_indices: np.ndarray,
    truth: np.ndarray,
    pools: list[int],
    batch_size: int = 128,
) -> list[dict[str, float]]:
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


def train_projection(
    rel: np.ndarray,
    train_queries: np.ndarray,
    train_truth: np.ndarray,
    train_negs: np.ndarray,
    proj_dim: int,
    steps: int,
    batch_size: int,
    neg_count: int,
    lr: float,
    temp: float,
    identity_reg: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sig = torch.from_numpy(rel.astype(np.float32)).to(device)
    if proj_dim == rel.shape[1]:
        projection = torch.eye(rel.shape[1], device=device, dtype=torch.float32)
        projection = projection + 0.001 * torch.randn_like(projection)
        identity_target = torch.eye(rel.shape[1], device=device, dtype=torch.float32)
    else:
        scale = 1.0 / np.sqrt(rel.shape[1])
        projection = torch.randn(rel.shape[1], proj_dim, device=device, dtype=torch.float32) * scale
        identity_target = None
    projection.requires_grad_(True)
    opt = torch.optim.AdamW([projection], lr=lr, weight_decay=1e-4)
    train_rows = np.arange(len(train_queries))
    losses = []

    for step in range(steps):
        rows = rng.choice(train_rows, size=min(batch_size, len(train_rows)), replace=False)
        q_idx = torch.as_tensor(train_queries[rows], dtype=torch.long, device=device)
        pos_np = np.array([rng.choice(train_truth[row]) for row in rows], dtype=np.int64)
        neg_np = np.vstack(
            [rng.choice(train_negs[row], size=neg_count, replace=len(train_negs[row]) < neg_count) for row in rows]
        )
        doc_np = np.concatenate([pos_np[:, None], neg_np], axis=1)
        doc_idx = torch.as_tensor(doc_np, dtype=torch.long, device=device)

        q = F.normalize(sig[q_idx] @ projection, dim=1)
        docs = F.normalize(torch.einsum("bkd,dr->bkr", sig[doc_idx], projection), dim=2)
        scores = torch.einsum("br,bkr->bk", q, docs) / temp
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
        loss = F.cross_entropy(scores, labels)
        if identity_target is not None and identity_reg > 0:
            loss = loss + identity_reg * F.mse_loss(projection, identity_target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    return projection.detach().cpu().numpy().astype(np.float32), losses


def train_ridge_projection(rel: np.ndarray, raw: np.ndarray, train_docs: np.ndarray, ridge_reg: float) -> np.ndarray:
    s = rel[train_docs].astype(np.float64)
    y = raw[train_docs].astype(np.float64)
    lhs = s.T @ s
    lhs += ridge_reg * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = s.T @ y
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pool(rows: list[dict[str, object]], path: Path, pool: int) -> None:
    selected = [r for r in rows if int(r["pool_size"]) == pool]
    labels = [str(r["method"]) for r in selected]
    vals = [float(r["mean_recall"]) for r in selected]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B"][: len(vals)])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recall of raw top-10")
    ax.set_title(f"Relation-only bilinear metric at pool {pool}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def write_markdown(path: Path, rows: list[dict[str, object]], losses: list[float], args: argparse.Namespace) -> None:
    pools = sorted({int(r["pool_size"]) for r in rows})
    methods = [m for m in ["cosine_relation", "contrastive_bilinear", "ridge_bilinear"] if any(r["method"] == m for r in rows)]
    table = []
    for method in methods:
        subset = {int(r["pool_size"]): r for r in rows if r["method"] == method}
        table.append([method, *[f"{float(subset[p]['mean_recall']):.4f}" for p in pools]])

    delta_rows = []
    cosine = {int(r["pool_size"]): r for r in rows if r["method"] == "cosine_relation"}
    bilinear = {int(r["pool_size"]): r for r in rows if r["method"] == "ridge_bilinear"}
    for pool in pools:
        delta_rows.append(
            [
                pool,
                f"{float(cosine[pool]['mean_recall']):.4f}",
                f"{float(bilinear[pool]['mean_recall']):.4f}" if pool in bilinear else "",
                f"{float(bilinear[pool]['mean_recall']) - float(cosine[pool]['mean_recall']):+.4f}" if pool in bilinear else "",
                f"{float(bilinear[pool]['all_top_k_contained']):.4f}" if pool in bilinear else "",
            ]
        )

    lines = [
        "# Bilinear Relation Metric",
        "",
        "## Scope",
        "",
        f"- Embedding file: `{args.embedding_path.as_posix()}`",
        f"- Documents: {args.n_docs}",
        f"- Train docs: first {args.train_docs}",
        f"- Eval queries: {args.eval_queries} sampled from held-out later docs",
        f"- Anchors: {args.anchor_count} farthest base-document anchors",
        f"- Projection rank: {args.proj_dim}",
        f"- Training steps: {args.steps}",
        "",
        "## Method",
        "",
        "Fixed relation signatures are computed once. The baseline ranks by cosine in signature space. `contrastive_bilinear` trains a projection `project(sig) = normalize(sig @ L)` with raw-neighbor positives and hard relation negatives. `ridge_bilinear` distills raw geometry by solving a ridge regression from relation signatures to raw embeddings, then ranks by cosine in the projected relation-only space. Both learned methods are bilinear at scoring time: `score(q,d) = sig(q)^T W sig(d)`. Evaluation is relation-only: no raw embedding rerank is used.",
        "",
        "## Recall by Candidate Count",
        "",
        *markdown_table(["method", *[f"pool {p}" for p in pools]], table),
        "",
        "## Delta vs Cosine",
        "",
        *markdown_table(["pool", "cosine", "bilinear", "delta", "bilinear all top-k"], delta_rows),
        "",
        "## Training",
        "",
        f"- Initial loss: {losses[0]:.4f}",
        f"- Final loss: {losses[-1]:.4f}",
        "",
        "## Finding",
        "",
        "This directly tests whether correlations between anchor relations recover final relation-only top-k. A positive delta at pool 10 means the bilinear metric improves final top-k ordering, while high recall only at larger pools means it remains a routing improvement.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{(path.parent / 'bilinear_relation_metric.csv').as_posix()}`",
        f"- Contrastive projection: `{(path.parent / 'contrastive_bilinear_projection.npy').as_posix()}`",
        f"- Ridge projection: `{(path.parent / 'ridge_bilinear_projection.npy').as_posix()}`",
        f"- Plot: `{(path.parent / 'bilinear_relation_pool10.png').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, default=Path("experiments/tenk_minilm_candidate/outputs/embeddings/sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/bilinear_relation_metric"))
    parser.add_argument("--n-docs", type=int, default=10000)
    parser.add_argument("--train-docs", type=int, default=8000)
    parser.add_argument("--train-queries", type=int, default=2000)
    parser.add_argument("--eval-queries", type=int, default=1000)
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pools", nargs="+", type=int, default=[10, 25, 50, 100, 250])
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--neg-count", type=int, default=64)
    parser.add_argument("--hard-pool", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--identity-reg", type=float, default=0.01)
    parser.add_argument("--ridge-reg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw = l2_normalize(np.load(args.embedding_path).astype(np.float32)[: args.n_docs])
    train_docs = np.arange(args.train_docs, dtype=np.int32)
    eval_docs = np.arange(args.train_docs, len(raw), dtype=np.int32)
    train_queries = np.sort(rng.choice(train_docs, size=min(args.train_queries, len(train_docs)), replace=False))
    eval_queries = np.sort(rng.choice(eval_docs, size=min(args.eval_queries, len(eval_docs)), replace=False))

    anchors = select_farthest_anchors(raw, args.anchor_count, args.seed, candidates=train_docs)
    rel = make_signatures(raw, anchors)
    train_truth = raw_truth(raw, train_queries, args.top_k, search_indices=train_docs)
    eval_truth = raw_truth(raw, eval_queries, args.top_k)
    train_negs = relation_hard_negatives(
        rel,
        train_queries,
        train_truth,
        args.neg_count,
        args.hard_pool,
        train_docs,
        np.random.default_rng(args.seed),
    )

    contrastive_projection, losses = train_projection(
        rel,
        train_queries,
        train_truth,
        train_negs,
        args.proj_dim,
        args.steps,
        args.batch_size,
        args.neg_count,
        args.lr,
        args.temp,
        args.identity_reg,
        args.seed,
        args.device,
    )
    contrastive_projected = l2_normalize(rel @ contrastive_projection)
    ridge_projection = train_ridge_projection(rel, raw, train_docs, args.ridge_reg)
    ridge_projected = l2_normalize(rel @ ridge_projection)

    rows = []
    for method, vectors, proj_dim in [
        ("cosine_relation", rel, 0),
        ("contrastive_bilinear", contrastive_projected, args.proj_dim),
        ("ridge_bilinear", ridge_projected, raw.shape[1]),
    ]:
        eval_rows = evaluate(vectors, eval_queries, eval_truth, args.pools, args.batch_size)
        for row in eval_rows:
            rows.append(
                {
                    "method": method,
                    "doc_count": len(raw),
                    "train_docs": args.train_docs,
                    "eval_queries": len(eval_queries),
                    "anchor_count": args.anchor_count,
                    "proj_dim": proj_dim,
                    "top_k": args.top_k,
                    **row,
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                }
            )

    np.save(args.out_dir / "contrastive_bilinear_projection.npy", contrastive_projection)
    np.save(args.out_dir / "ridge_bilinear_projection.npy", ridge_projection)
    write_csv(args.out_dir / "bilinear_relation_metric.csv", rows)
    plot_pool(rows, args.out_dir / "bilinear_relation_pool10.png", 10 if 10 in args.pools else args.pools[0])
    write_markdown(args.out_dir / "bilinear_relation_metric.md", rows, losses, args)
    print(f"Wrote {args.out_dir / 'bilinear_relation_metric.md'}")


if __name__ == "__main__":
    main()
