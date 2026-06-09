import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.argsort(-scores)
    part = np.argpartition(-scores, kth=k - 1)[:k]
    return part[np.argsort(-scores[part])]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class HFMeanPoolEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        max_tokens: int = 160,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            emb = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings.append(torch.nn.functional.normalize(emb.float(), p=2, dim=1).cpu().numpy())
        return np.vstack(embeddings).astype(np.float32)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    relative_score: float
    index: int


class LARPIndex:
    """Fixed-anchor relative-signature candidate index with raw embedding rerank."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        anchor_count: int = 256,
        signature_transform: str = "row_zscore",
        seed: int = 7,
    ):
        self.model_name = model_name
        self.anchor_count = anchor_count
        self.signature_transform = signature_transform
        self.seed = seed
        self.doc_ids: list[str] = []
        self.raw_embeddings: np.ndarray | None = None
        self.relative_signatures: np.ndarray | None = None
        self.anchor_indices: np.ndarray | None = None
        self.anchor_embeddings: np.ndarray | None = None

    @staticmethod
    def select_farthest_anchors(embeddings: np.ndarray, count: int, seed: int = 7) -> np.ndarray:
        raw = l2_normalize(embeddings.astype(np.float32))
        rng = np.random.default_rng(seed)
        first = int(rng.integers(0, len(raw)))
        selected = [first]
        max_sim = raw @ raw[first]
        for _ in range(1, min(count, len(raw))):
            next_idx = int(np.argmin(max_sim))
            selected.append(next_idx)
            max_sim = np.maximum(max_sim, raw @ raw[next_idx])
        return np.array(selected, dtype=np.int32)

    @staticmethod
    def transform_signatures(signatures: np.ndarray, transform: str) -> np.ndarray:
        x = signatures.astype(np.float32)
        if transform == "raw":
            return l2_normalize(x)
        if transform == "row_zscore":
            centered = x - x.mean(axis=1, keepdims=True)
            scaled = centered / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
            return l2_normalize(scaled)
        if transform == "rank":
            ranks = np.empty_like(x, dtype=np.float32)
            order = np.argsort(-x, axis=1)
            ranks[np.arange(x.shape[0])[:, None], order] = np.arange(x.shape[1], dtype=np.float32)
            return l2_normalize(1.0 - ranks / max(1, x.shape[1] - 1))
        raise ValueError(f"Unsupported signature transform: {transform}")

    def _signature_from_raw(self, raw_embeddings: np.ndarray) -> np.ndarray:
        if self.anchor_embeddings is None:
            raise RuntimeError("Index has no anchor embeddings.")
        raw = l2_normalize(raw_embeddings.astype(np.float32))
        sig = raw @ self.anchor_embeddings.T
        return self.transform_signatures(sig, self.signature_transform)

    def fit_embeddings(self, embeddings: np.ndarray, doc_ids: Iterable[str] | None = None) -> "LARPIndex":
        raw = l2_normalize(embeddings.astype(np.float32))
        self.doc_ids = list(doc_ids) if doc_ids is not None else [str(i) for i in range(len(raw))]
        if len(self.doc_ids) != len(raw):
            raise ValueError("doc_ids length must match embeddings length.")
        self.raw_embeddings = raw
        self.anchor_indices = self.select_farthest_anchors(raw, self.anchor_count, self.seed)
        self.anchor_embeddings = raw[self.anchor_indices]
        self.relative_signatures = self._signature_from_raw(raw)
        return self

    def fit_texts(self, texts: list[str], doc_ids: Iterable[str] | None = None, batch_size: int = 32) -> "LARPIndex":
        embedder = HFMeanPoolEmbedder(self.model_name)
        return self.fit_embeddings(embedder.encode(texts, batch_size=batch_size), doc_ids)

    def insert_embedding(self, embedding: np.ndarray, doc_id: str) -> None:
        self.insert_embeddings(np.asarray(embedding, dtype=np.float32).reshape(1, -1), [doc_id])

    def insert_embeddings(self, embeddings: np.ndarray, doc_ids: Iterable[str]) -> None:
        doc_ids = [str(doc_id) for doc_id in doc_ids]
        raw = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        if len(raw) != len(doc_ids):
            raise ValueError("doc_ids length must match embeddings length.")
        sig = self._signature_from_raw(raw)
        self.raw_embeddings = raw if self.raw_embeddings is None else np.vstack([self.raw_embeddings, raw])
        self.relative_signatures = sig if self.relative_signatures is None else np.vstack([self.relative_signatures, sig])
        self.doc_ids.extend(doc_ids)

    def insert_text(self, text: str, doc_id: str, embedder: HFMeanPoolEmbedder | None = None) -> None:
        embedder = embedder or HFMeanPoolEmbedder(self.model_name)
        self.insert_embedding(embedder.encode([text])[0], doc_id)

    def search_embedding(self, query_embedding: np.ndarray, top_k: int = 10, pool: int = 100) -> list[SearchResult]:
        if self.raw_embeddings is None or self.relative_signatures is None:
            raise RuntimeError("Index is empty.")
        query_raw = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        query_sig = self._signature_from_raw(query_raw)[0]
        rel_scores = self.relative_signatures @ query_sig
        pool_indices = top_indices(rel_scores, min(pool, len(rel_scores)))
        raw_scores = self.raw_embeddings[pool_indices] @ query_raw[0]
        reranked = pool_indices[top_indices(raw_scores, min(top_k, len(pool_indices)))]
        return [
            SearchResult(
                doc_id=self.doc_ids[int(idx)],
                score=float(self.raw_embeddings[int(idx)] @ query_raw[0]),
                relative_score=float(rel_scores[int(idx)]),
                index=int(idx),
            )
            for idx in reranked
        ]

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        pool: int = 100,
        embedder: HFMeanPoolEmbedder | None = None,
    ) -> list[SearchResult]:
        embedder = embedder or HFMeanPoolEmbedder(self.model_name)
        return self.search_embedding(embedder.encode([query])[0], top_k=top_k, pool=pool)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_name": self.model_name,
            "anchor_count": self.anchor_count,
            "signature_transform": self.signature_transform,
            "seed": self.seed,
            "doc_ids": self.doc_ids,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.save(path / "raw_embeddings.npy", self.raw_embeddings)
        np.save(path / "relative_signatures.npy", self.relative_signatures)
        np.save(path / "anchor_indices.npy", self.anchor_indices)
        np.save(path / "anchor_embeddings.npy", self.anchor_embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "LARPIndex":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        index = cls(
            model_name=metadata["model_name"],
            anchor_count=metadata["anchor_count"],
            signature_transform=metadata["signature_transform"],
            seed=metadata["seed"],
        )
        index.doc_ids = list(metadata["doc_ids"])
        index.raw_embeddings = np.load(path / "raw_embeddings.npy")
        index.relative_signatures = np.load(path / "relative_signatures.npy")
        index.anchor_indices = np.load(path / "anchor_indices.npy")
        index.anchor_embeddings = np.load(path / "anchor_embeddings.npy")
        return index

    def benchmark_queries(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        pool: int = 100,
    ) -> dict[str, float]:
        start = time.perf_counter()
        for embedding in query_embeddings:
            self.search_embedding(embedding, top_k=top_k, pool=pool)
        elapsed = time.perf_counter() - start
        return {
            "query_count": len(query_embeddings),
            "total_seconds": elapsed,
            "ms_per_query": 1000.0 * elapsed / max(1, len(query_embeddings)),
        }
