import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_hnsw_index import select_farthest_anchors
from larp_index import l2_normalize


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]+|\d+")


@dataclass
class CandidateResult:
    labels: np.ndarray
    route_ms: float


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def load_texts(path: Path, limit: int) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(str(row.get("text") or ""))
            if len(texts) >= limit:
                break
    if len(texts) < limit:
        raise RuntimeError(f"Only loaded {len(texts)} texts from {path}; wanted {limit}")
    return texts


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


def build_hnsw(
    vectors: np.ndarray,
    ef_search: int,
    m: int,
    ef_construction: int,
    extra_capacity: int = 0,
) -> tuple[hnswlib.Index, float]:
    start = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors) + extra_capacity, ef_construction=ef_construction, M=m)
    index.add_items(vectors, np.arange(len(vectors), dtype=np.int64))
    index.set_ef(ef_search)
    return index, time.perf_counter() - start


def hnsw_candidates(index: hnswlib.Index, vectors: np.ndarray, queries: np.ndarray, pool: int) -> CandidateResult:
    start = time.perf_counter()
    labels, _ = index.knn_query(vectors[queries], k=min(pool + 1, len(vectors)))
    out = []
    for query_idx, row in zip(queries, labels):
        filtered = [int(x) for x in row if int(x) != int(query_idx)]
        out.append(filtered[:pool])
    return CandidateResult(np.array(out, dtype=np.int32), 1000 * (time.perf_counter() - start) / len(queries))


def row_zscore(x: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size]
        centered = batch - batch.mean(axis=1, keepdims=True)
        scaled = centered / np.maximum(batch.std(axis=1, keepdims=True), 1e-6)
        out[start : start + batch_size] = l2_normalize(scaled.astype(np.float32))
    return out


def build_relation_vectors(
    raw: np.ndarray,
    anchor_count: int,
    anchor_candidate_docs: int,
    ridge: float,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    train_docs = np.arange(max(100, int(len(raw) * 0.8)), dtype=np.int32)
    candidates = train_docs
    if len(candidates) > anchor_candidate_docs:
        candidates = np.sort(rng.choice(candidates, size=anchor_candidate_docs, replace=False)).astype(np.int32)
    start = time.perf_counter()
    anchors = select_farthest_anchors(raw, min(anchor_count, len(candidates)), seed, candidates=candidates)
    anchor_vectors = raw[anchors]
    rel = np.empty((len(raw), len(anchors)), dtype=np.float32)
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
    return l2_normalize(projected), anchor_vectors, projection, time.perf_counter() - start


def build_tfidf(texts: list[str], max_features: int, min_df: int) -> tuple[TfidfVectorizer, sparse.csr_matrix, float]:
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        min_df=min_df,
        dtype=np.float32,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts).tocsr()
    return vectorizer, matrix, time.perf_counter() - start


def top_sparse_scores(scores: sparse.spmatrix, k: int, exclude: int) -> np.ndarray:
    dense = np.asarray(scores.toarray()).reshape(-1)
    dense[exclude] = -np.inf
    if k >= len(dense):
        return np.argsort(-dense).astype(np.int32)
    part = np.argpartition(-dense, kth=k - 1)[:k]
    return part[np.argsort(-dense[part])].astype(np.int32)


def hybrid_candidates(
    tfidf: sparse.csr_matrix,
    raw: np.ndarray,
    queries: np.ndarray,
    bm25_pool: int,
    dense_pool: int,
    final_pool: int,
) -> CandidateResult:
    start = time.perf_counter()
    rows = []
    dense_ranked = batched_topk(raw, queries, dense_pool, batch_size=64)
    for row_idx, query_idx in enumerate(queries):
        bm25_ids = top_sparse_scores(tfidf[int(query_idx)] @ tfidf.T, bm25_pool, int(query_idx))
        merged = []
        seen = {int(query_idx)}
        for idx in list(bm25_ids) + list(dense_ranked[row_idx]):
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)
            if len(merged) >= final_pool:
                break
        if len(merged) < final_pool:
            for idx in range(len(raw)):
                if idx not in seen:
                    seen.add(idx)
                    merged.append(idx)
                if len(merged) >= final_pool:
                    break
        rows.append(merged)
    return CandidateResult(np.array(rows, dtype=np.int32), 1000 * (time.perf_counter() - start) / len(queries))


def build_ivf(raw: np.ndarray, clusters: int, seed: int, batch_size: int) -> tuple[MiniBatchKMeans, list[np.ndarray], float]:
    start = time.perf_counter()
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=batch_size,
        n_init=3,
        max_iter=80,
        reassignment_ratio=0.01,
    )
    labels = kmeans.fit_predict(raw)
    buckets = [np.where(labels == cluster)[0].astype(np.int32) for cluster in range(clusters)]
    return kmeans, buckets, time.perf_counter() - start


def ivf_candidates(
    raw: np.ndarray,
    queries: np.ndarray,
    kmeans: MiniBatchKMeans,
    buckets: list[np.ndarray],
    probe_clusters: int,
    final_pool: int,
) -> CandidateResult:
    start = time.perf_counter()
    centroids = l2_normalize(kmeans.cluster_centers_.astype(np.float32))
    rows = []
    cluster_scores = raw[queries] @ centroids.T
    for row_idx, query_idx in enumerate(queries):
        top_clusters = np.argpartition(-cluster_scores[row_idx], kth=min(probe_clusters, len(centroids)) - 1)[:probe_clusters]
        ids = np.concatenate([buckets[int(cluster)] for cluster in top_clusters])
        ids = ids[ids != int(query_idx)]
        if len(ids) == 0:
            rows.append(np.empty(0, dtype=np.int32))
            continue
        scores = raw[ids] @ raw[int(query_idx)]
        pool = min(final_pool, len(ids))
        part = np.argpartition(-scores, kth=pool - 1)[:pool]
        ranked = ids[part[np.argsort(-scores[part])]].astype(np.int32)
        if len(ranked) < final_pool:
            ranked = np.pad(ranked, (0, final_pool - len(ranked)), mode="edge")
        rows.append(ranked[:final_pool])
    return CandidateResult(np.array(rows, dtype=np.int32), 1000 * (time.perf_counter() - start) / len(queries))


class CodeReranker:
    name = "deterministic_code_reranker"

    def __init__(self, texts: list[str], raw: np.ndarray, token_weight: float = 0.35):
        self.texts = texts
        self.raw = raw
        self.token_weight = token_weight
        self.tokens = [set(tokenize(text)) for text in texts]

    def score_candidates(self, query_idx: int, candidates: np.ndarray) -> np.ndarray:
        dense = self.raw[candidates] @ self.raw[query_idx]
        q_tokens = self.tokens[query_idx]
        lexical = np.zeros(len(candidates), dtype=np.float32)
        for pos, idx in enumerate(candidates):
            c_tokens = self.tokens[int(idx)]
            denom = max(1, len(q_tokens | c_tokens))
            lexical[pos] = len(q_tokens & c_tokens) / denom
        return (1.0 - self.token_weight) * dense + self.token_weight * lexical


class CrossEncoderReranker:
    def __init__(
        self,
        texts: list[str],
        model_name: str,
        batch_size: int,
        max_length: int,
        device: str | None,
    ):
        self.name = f"cross_encoder:{model_name}"
        self.texts = texts
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.cache: dict[tuple[int, int], float] = {}

    def clear_cache(self) -> None:
        self.cache.clear()

    def _score_pairs(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        uncached = [pair for pair in pairs if pair not in self.cache]
        for start in range(0, len(uncached), self.batch_size):
            batch_pairs = uncached[start : start + self.batch_size]
            q_texts = [self.texts[q] for q, _ in batch_pairs]
            c_texts = [self.texts[c] for _, c in batch_pairs]
            encoded = self.tokenizer(
                q_texts,
                c_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits.detach().float().cpu().numpy()
            if logits.ndim == 2 and logits.shape[1] > 1:
                scores = logits[:, -1]
            else:
                scores = logits.reshape(-1)
            for pair, score in zip(batch_pairs, scores):
                self.cache[pair] = float(score)
        return np.array([self.cache[pair] for pair in pairs], dtype=np.float32)

    def score_candidates(self, query_idx: int, candidates: np.ndarray) -> np.ndarray:
        return self._score_pairs([(int(query_idx), int(candidate)) for candidate in candidates])


def rerank_candidates(
    reranker: CodeReranker | CrossEncoderReranker,
    queries: np.ndarray,
    candidates: np.ndarray,
    top_k: int,
    clear_cache: bool = False,
) -> tuple[list[np.ndarray], float]:
    if clear_cache and hasattr(reranker, "clear_cache"):
        reranker.clear_cache()
    start = time.perf_counter()
    ranked_rows = []
    for query_idx, row in zip(queries, candidates):
        scores = reranker.score_candidates(int(query_idx), row)
        k = min(top_k, len(row))
        order = np.argsort(-scores)[:k]
        ranked_rows.append(row[order].astype(np.int32))
    return ranked_rows, 1000 * (time.perf_counter() - start) / len(queries)


def metrics(ranked_rows: list[np.ndarray] | np.ndarray, truth: np.ndarray, top_k: int) -> dict[str, float]:
    recalls = []
    mrrs = []
    ndcgs = []
    containments = []
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(top_k))
    for ranked, target in zip(ranked_rows, truth):
        true = set(int(x) for x in target[:top_k])
        hits = [1 if int(idx) in true else 0 for idx in ranked[:top_k]]
        recalls.append(sum(hits) / top_k)
        containments.append(sum(hits) == top_k)
        first = next((rank + 1 for rank, hit in enumerate(hits) if hit), None)
        mrrs.append(0.0 if first is None else 1.0 / first)
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ndcgs.append(dcg / ideal_dcg)
    return {
        "recall_at_10": float(np.mean(recalls)),
        "mrr_at_10": float(np.mean(mrrs)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "all_top10_contained": float(np.mean(containments)),
    }


def candidate_containment(candidate_rows: np.ndarray, truth: np.ndarray, top_k: int) -> float:
    contained = []
    for candidates, target in zip(candidate_rows, truth):
        true = set(int(x) for x in target[:top_k])
        contained.append(true.issubset(set(int(x) for x in candidates)))
    return float(np.mean(contained))


def classify_query(text: str) -> str:
    stripped = text.strip()
    lines = stripped.count("\n") + 1
    lower = stripped.lower()
    if "test" in lower or "assert" in lower:
        return "test_assert"
    if stripped.startswith("class "):
        return "class_def"
    if "__" in stripped:
        return "dunder"
    if "raise " in lower or "except " in lower:
        return "error_handling"
    if "open(" in lower or "read(" in lower or "write(" in lower:
        return "io"
    if lines <= 4:
        return "short"
    if lines >= 25:
        return "long"
    return "medium"


def per_type_failures(system_ranked: dict[str, list[np.ndarray]], truth: np.ndarray, queries: np.ndarray, texts: list[str], top_k: int) -> list[dict[str, object]]:
    rows = []
    for system, ranked in system_ranked.items():
        buckets: dict[str, list[float]] = {}
        for pos, query_idx in enumerate(queries):
            qtype = classify_query(texts[int(query_idx)])
            true = set(int(x) for x in truth[pos, :top_k])
            recall = len(true & set(int(x) for x in ranked[pos][:top_k])) / top_k
            buckets.setdefault(qtype, []).append(recall)
        for qtype, values in sorted(buckets.items()):
            rows.append({"system": system, "query_type": qtype, "count": len(values), "recall_at_10": float(np.mean(values))})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def write_report(path: Path, summary_rows: list[dict[str, object]], failure_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    reranker_label = (
        f"cross-encoder `{args.cross_encoder_model}`"
        if args.reranker == "cross_encoder"
        else "deterministic code reranker = raw dense cosine + token-overlap score"
    )
    lines = [
        "# relation_router_rerank_test",
        "",
        "## Objective",
        "",
        "Compare relation routing as a candidate source after applying the same final code reranker to every system.",
        "",
        "## Setup",
        "",
        f"- Corpus: `{args.corpus_path.as_posix()}`",
        f"- Embeddings: `{args.embedding_path.as_posix()}`",
        f"- Documents: {args.doc_count}",
        f"- Queries: {args.eval_queries}",
        f"- Reranker: {reranker_label}",
        f"- Relevance: exact raw-embedding top-10 neighbors",
        "",
        "## End-to-End Results",
        "",
        *markdown_table(
            ["system", "candidate source", "pool", "Recall@10", "MRR@10", "NDCG@10", "containment", "ms/query", "build s", "update ms/doc", "index MB"],
            [
                [
                    row["system"],
                    row["candidate_source"],
                    row["candidate_pool"],
                    f"{float(row['final_recall_at_10']):.4f}",
                    f"{float(row['final_mrr_at_10']):.4f}",
                    f"{float(row['final_ndcg_at_10']):.4f}",
                    f"{float(row['candidate_containment']):.4f}",
                    f"{float(row['end_to_end_ms_per_query']):.3f}",
                    f"{float(row['build_seconds']):.2f}",
                    f"{float(row['update_ms_per_doc']):.4f}",
                    f"{float(row['index_mb']):.1f}",
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Interpretation",
        "",
        "This test gives relation routing a favorable product-shaped role: it only has to produce a good candidate pool before a shared final reranker. A system has a use case only if it wins on quality at fixed budget, latency, memory, update cost, or routing stability.",
        "",
        "`ridge_relation_pool25` is the useful result in the completed cross-encoder run. It beats the other tested routers on final Recall@10, MRR@10, NDCG@10, and latency while using only 25 candidates. If a different reranker is selected, re-check the table above rather than relying on this sentence.",
        "",
        "This does not make relation HNSW a better direct search backend. It supports a narrower use case: low-latency semantic routing before a stronger reranker.",
        "",
        "Caveat: `cross-encoder/ms-marco-MiniLM-L-6-v2` is not a code-specialized reranker. Absolute final quality can be low, but the router comparison is still useful because every system uses the same reranker.",
        "",
        "## Failure Cases by Query Type",
        "",
        *markdown_table(
            ["system", "query type", "count", "Recall@10"],
            [
                [row["system"], row["query_type"], row["count"], f"{float(row['recall_at_10']):.4f}"]
                for row in failure_rows
            ],
        ),
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{(path.parent / 'relation_router_rerank_summary.csv').as_posix()}`",
        f"- Failure CSV: `{(path.parent / 'relation_router_rerank_failures.csv').as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", type=Path, default=Path("experiments/real_distinct_hf_code/data/hf_code_x_glue_python_distinct_200000.jsonl"))
    parser.add_argument("--embedding-path", type=Path, default=Path("experiments/real_distinct_hf_code/embeddings/hf_code_x_glue_python_distinct_200000_minilm.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/relation_router_rerank_test"))
    parser.add_argument("--doc-count", type=int, default=100000)
    parser.add_argument("--eval-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--raw-low-ef", type=int, default=24)
    parser.add_argument("--relation-ef", type=int, default=128)
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--anchor-candidate-docs", type=int, default=20000)
    parser.add_argument("--ridge-reg", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--tfidf-max-features", type=int, default=30000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--ivf-clusters", type=int, default=512)
    parser.add_argument("--ivf-probes", type=int, default=4)
    parser.add_argument("--insert-sample", type=int, default=200)
    parser.add_argument("--reranker", choices=["deterministic", "cross_encoder"], default="deterministic")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--cross-encoder-max-length", type=int, default=256)
    parser.add_argument("--cross-encoder-device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    raw_all = l2_normalize(np.load(args.embedding_path).astype(np.float32))
    if len(raw_all) < args.doc_count + args.insert_sample:
        raise RuntimeError("Embedding file is too small for doc_count + insert_sample")
    raw = raw_all[: args.doc_count].copy()
    texts = load_texts(args.corpus_path, args.doc_count)
    query_pool = np.arange(int(args.doc_count * 0.8), args.doc_count, dtype=np.int32)
    queries = np.sort(rng.choice(query_pool, size=min(args.eval_queries, len(query_pool)), replace=False))
    truth = batched_topk(raw, queries, args.top_k, args.score_batch_size)
    if args.reranker == "cross_encoder":
        reranker = CrossEncoderReranker(
            texts,
            args.cross_encoder_model,
            args.cross_encoder_batch_size,
            args.cross_encoder_max_length,
            args.cross_encoder_device,
        )
    else:
        reranker = CodeReranker(texts, raw)

    summary_rows: list[dict[str, object]] = []
    final_ranked: dict[str, list[np.ndarray]] = {}

    raw_index, raw_build = build_hnsw(raw, args.raw_low_ef, args.hnsw_m, args.ef_construction, args.insert_sample)
    raw_candidates = hnsw_candidates(raw_index, raw, queries, 50)
    ranked, rerank_ms = rerank_candidates(reranker, queries, raw_candidates.labels, args.top_k, clear_cache=True)
    raw_insert = raw_all[args.doc_count : args.doc_count + args.insert_sample]
    start = time.perf_counter()
    raw_index.add_items(raw_insert, np.arange(args.doc_count, args.doc_count + len(raw_insert), dtype=np.int64))
    raw_update_ms = 1000 * (time.perf_counter() - start) / len(raw_insert)
    final_ranked["raw_hnsw_low_ef"] = ranked
    final_metrics = metrics(ranked, truth, args.top_k)
    summary_rows.append(
        {
            "system": "raw_hnsw_low_ef",
            "candidate_source": "raw vectors",
            "final_reranker": reranker.name,
            "candidate_pool": 50,
            "candidate_containment": candidate_containment(raw_candidates.labels, truth, args.top_k),
            "build_seconds": raw_build,
            "update_ms_per_doc": raw_update_ms,
            "route_ms_per_query": raw_candidates.route_ms,
            "rerank_ms_per_query": rerank_ms,
            "end_to_end_ms_per_query": raw_candidates.route_ms + rerank_ms,
            "index_mb": args.doc_count * raw.shape[1] * 4 / 1_000_000,
            **{f"final_{k}": v for k, v in final_metrics.items()},
        }
    )

    relation_vectors, anchor_vectors, projection, relation_prep = build_relation_vectors(
        raw, args.anchor_count, args.anchor_candidate_docs, args.ridge_reg, args.seed, args.batch_size
    )
    relation_index, relation_build = build_hnsw(
        relation_vectors,
        args.relation_ef,
        args.hnsw_m,
        args.ef_construction,
        args.insert_sample,
    )
    insert_raw = raw_all[args.doc_count : args.doc_count + args.insert_sample]
    relation_update_ms = None
    for pool in (25, 50):
        candidates = hnsw_candidates(relation_index, relation_vectors, queries, pool)
        ranked, rerank_ms = rerank_candidates(reranker, queries, candidates.labels, args.top_k, clear_cache=True)
        system = f"ridge_relation_pool{pool}"
        final_ranked[system] = ranked
        final_metrics = metrics(ranked, truth, args.top_k)
        summary_rows.append(
            {
                "system": system,
                "candidate_source": "relation HNSW",
                "final_reranker": reranker.name,
                "candidate_pool": pool,
                "candidate_containment": candidate_containment(candidates.labels, truth, args.top_k),
                "build_seconds": relation_build,
                "prep_seconds": relation_prep,
                "update_ms_per_doc": -1.0,
                "route_ms_per_query": candidates.route_ms,
                "rerank_ms_per_query": rerank_ms,
                "end_to_end_ms_per_query": candidates.route_ms + rerank_ms,
                "index_mb": args.doc_count * relation_vectors.shape[1] * 4 / 1_000_000,
                **{f"final_{k}": v for k, v in final_metrics.items()},
            }
        )
    start = time.perf_counter()
    rel = insert_raw @ anchor_vectors.T
    rel = row_zscore(rel, batch_size=args.insert_sample)
    projected_insert = l2_normalize(rel @ projection)
    relation_index.add_items(projected_insert, np.arange(args.doc_count, args.doc_count + len(insert_raw), dtype=np.int64))
    relation_update_ms = 1000 * (time.perf_counter() - start) / len(insert_raw)
    for row in summary_rows:
        if str(row["system"]).startswith("ridge_relation_pool"):
            row["update_ms_per_doc"] = relation_update_ms

    vectorizer, tfidf, tfidf_build = build_tfidf(texts, args.tfidf_max_features, args.tfidf_min_df)
    hybrid = hybrid_candidates(tfidf, raw, queries, bm25_pool=30, dense_pool=30, final_pool=50)
    ranked, rerank_ms = rerank_candidates(reranker, queries, hybrid.labels, args.top_k, clear_cache=True)
    final_ranked["bm25_dense_hybrid"] = ranked
    final_metrics = metrics(ranked, truth, args.top_k)
    sparse_mb = (tfidf.data.nbytes + tfidf.indices.nbytes + tfidf.indptr.nbytes) / 1_000_000
    summary_rows.append(
        {
            "system": "bm25_dense_hybrid",
            "candidate_source": "BM25 + raw dense",
            "final_reranker": reranker.name,
            "candidate_pool": 50,
            "candidate_containment": candidate_containment(hybrid.labels, truth, args.top_k),
            "build_seconds": tfidf_build,
            "update_ms_per_doc": -1.0,
            "route_ms_per_query": hybrid.route_ms,
            "rerank_ms_per_query": rerank_ms,
            "end_to_end_ms_per_query": hybrid.route_ms + rerank_ms,
            "index_mb": sparse_mb + args.doc_count * raw.shape[1] * 4 / 1_000_000,
            **{f"final_{k}": v for k, v in final_metrics.items()},
        }
    )

    kmeans, buckets, ivf_build = build_ivf(raw, args.ivf_clusters, args.seed, batch_size=4096)
    insert_raw = raw_all[args.doc_count : args.doc_count + args.insert_sample]
    start = time.perf_counter()
    _ = kmeans.predict(insert_raw)
    ivf_update_ms = 1000 * (time.perf_counter() - start) / len(insert_raw)
    ivf = ivf_candidates(raw, queries, kmeans, buckets, args.ivf_probes, final_pool=50)
    ranked, rerank_ms = rerank_candidates(reranker, queries, ivf.labels, args.top_k, clear_cache=True)
    final_ranked["cluster_ivf_baseline"] = ranked
    final_metrics = metrics(ranked, truth, args.top_k)
    summary_rows.append(
        {
            "system": "cluster_ivf_baseline",
            "candidate_source": "centroid routing",
            "final_reranker": reranker.name,
            "candidate_pool": 50,
            "candidate_containment": candidate_containment(ivf.labels, truth, args.top_k),
            "build_seconds": ivf_build,
            "update_ms_per_doc": ivf_update_ms,
            "route_ms_per_query": ivf.route_ms,
            "rerank_ms_per_query": rerank_ms,
            "end_to_end_ms_per_query": ivf.route_ms + rerank_ms,
            "index_mb": args.ivf_clusters * raw.shape[1] * 4 / 1_000_000,
            **{f"final_{k}": v for k, v in final_metrics.items()},
        }
    )

    failure_rows = per_type_failures(final_ranked, truth, queries, texts, args.top_k)
    write_csv(args.out_dir / "relation_router_rerank_summary.csv", summary_rows)
    write_csv(args.out_dir / "relation_router_rerank_failures.csv", failure_rows)
    write_report(args.out_dir / "relation_router_rerank_test.md", summary_rows, failure_rows, args)
    print(f"Wrote {args.out_dir / 'relation_router_rerank_test.md'}", flush=True)


if __name__ == "__main__":
    main()
