import argparse
import csv
import json
import math
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_hnsw_index import LARPHNSWIndex
from larp_index import HFMeanPoolEmbedder, l2_normalize, top_indices


SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download_scifact(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "scifact.zip"
    extract_dir = out_dir / "scifact"
    if not zip_path.exists():
        print(f"Downloading {SCIFACT_URL} -> {zip_path}")
        urllib.request.urlretrieve(SCIFACT_URL, zip_path)
    if not (extract_dir / "corpus.jsonl").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    return extract_dir


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            docid = parts[1]
            score = int(parts[2])
            qrels[qid][docid] = score
    return qrels


def dcg(rels: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def metrics(qrels: dict[str, dict[str, int]], ranked: dict[str, list[str]], k: int) -> dict[str, float]:
    recalls = []
    mrrs = []
    ndcgs = []
    for qid, rels in qrels.items():
        relevant = {docid for docid, score in rels.items() if score > 0}
        if not relevant:
            continue
        pred = ranked.get(qid, [])[:k]
        hits = [1 if docid in relevant else 0 for docid in pred]
        recalls.append(sum(hits) / len(relevant))
        rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        gains = [rels.get(docid, 0) for docid in pred]
        ideal = sorted([score for score in rels.values() if score > 0], reverse=True)[:k]
        denom = dcg(ideal)
        ndcgs.append(dcg(gains) / denom if denom else 0.0)
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"mrr@{k}": float(np.mean(mrrs)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "query_count": len(recalls),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def raw_exact_ranked(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, qids: list[str], doc_ids: list[str], k: int) -> dict[str, list[str]]:
    ranked = {}
    for qid, query in zip(qids, query_embeddings):
        scores = doc_embeddings @ query
        ranked[qid] = [doc_ids[int(idx)] for idx in top_indices(scores, k)]
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/external/beir"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/beir_scifact"))
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--pools", nargs="+", type=int, default=[100, 250, 500])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = download_scifact(args.data_dir)
    corpus_rows = read_jsonl(data_dir / "corpus.jsonl")
    query_rows = read_jsonl(data_dir / "queries.jsonl")
    qrels_path = data_dir / "qrels" / "test.tsv"
    qrels = read_qrels(qrels_path)
    query_rows = [row for row in query_rows if str(row["_id"]) in qrels]

    doc_ids = [str(row["_id"]) for row in corpus_rows]
    doc_texts = [f"{row.get('title', '')}\n{row.get('text', '')}".strip() for row in corpus_rows]
    qids = [str(row["_id"]) for row in query_rows]
    query_texts = [str(row.get("text", "")) for row in query_rows]

    doc_emb_path = args.out_dir / "scifact_doc_embeddings.npy"
    query_emb_path = args.out_dir / "scifact_query_embeddings.npy"
    if doc_emb_path.exists() and query_emb_path.exists():
        doc_embeddings = np.load(doc_emb_path)
        query_embeddings = np.load(query_emb_path)
    else:
        embedder = HFMeanPoolEmbedder(args.model_name, max_tokens=args.max_tokens)
        doc_embeddings = embedder.encode(doc_texts, batch_size=args.batch_size)
        query_embeddings = embedder.encode(query_texts, batch_size=args.batch_size)
        np.save(doc_emb_path, doc_embeddings)
        np.save(query_emb_path, query_embeddings)

    doc_embeddings = l2_normalize(doc_embeddings.astype(np.float32))
    query_embeddings = l2_normalize(query_embeddings.astype(np.float32))
    index = LARPHNSWIndex(model_name=args.model_name, anchor_count=args.anchor_count, seed=args.seed)
    index.fit_embeddings(doc_embeddings, doc_ids=doc_ids)

    rows = []
    ranked_raw_exact = raw_exact_ranked(doc_embeddings, query_embeddings, qids, doc_ids, args.top_k)
    rows.append({"method": "raw_exact", "pool": args.top_k, **metrics(qrels, ranked_raw_exact, args.top_k)})

    ranked_raw_hnsw = {
        qid: [r.doc_id for r in index.search_raw_hnsw(query, top_k=args.top_k)]
        for qid, query in zip(qids, query_embeddings)
    }
    rows.append({"method": "raw_hnsw", "pool": args.top_k, **metrics(qrels, ranked_raw_hnsw, args.top_k)})

    ranked_relation = {
        qid: [r.doc_id for r in index.search_relative_only(query, top_k=args.top_k)]
        for qid, query in zip(qids, query_embeddings)
    }
    rows.append({"method": "larp_relation_only", "pool": args.top_k, **metrics(qrels, ranked_relation, args.top_k)})

    for pool in args.pools:
        ranked_larp = {
            qid: [r.doc_id for r in index.search_embedding(query, top_k=args.top_k, pool=pool)]
            for qid, query in zip(qids, query_embeddings)
        }
        rows.append({"method": "larp_hnsw_rerank", "pool": pool, **metrics(qrels, ranked_larp, args.top_k)})

    write_csv(args.out_dir / "beir_scifact_benchmark.csv", rows)
    lines = [
        "# BEIR SciFact Benchmark",
        "",
        f"- Corpus documents: {len(doc_ids)}",
        f"- Test queries with qrels: {len(qids)}",
        f"- Model: `{args.model_name}`",
        f"- Anchor count: {args.anchor_count}",
        "",
        *markdown_table(
            ["method", "pool", f"recall@{args.top_k}", f"mrr@{args.top_k}", f"ndcg@{args.top_k}", "queries"],
            [
                [
                    row["method"],
                    row["pool"],
                    f"{row[f'recall@{args.top_k}']:.4f}",
                    f"{row[f'mrr@{args.top_k}']:.4f}",
                    f"{row[f'ndcg@{args.top_k}']:.4f}",
                    row["query_count"],
                ]
                for row in rows
            ],
        ),
        "",
        "## Finding",
        "",
        "This is a real labeled retrieval benchmark, not a raw-neighbor proxy. If LARP reranking matches raw HNSW here, the routing layer preserves task relevance well enough for this small benchmark. If relation-only lags, reranking remains necessary.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{(args.out_dir / 'beir_scifact_benchmark.csv').as_posix()}`",
        f"- Doc embeddings: `{doc_emb_path.as_posix()}`",
        f"- Query embeddings: `{query_emb_path.as_posix()}`",
    ]
    (args.out_dir / "beir_scifact_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'beir_scifact_benchmark.md'}")


if __name__ == "__main__":
    main()
