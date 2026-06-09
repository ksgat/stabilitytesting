import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hnswlib
import numpy as np

from larp_index import HFMeanPoolEmbedder, SearchResult, l2_normalize, top_indices


def row_zscore(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    return l2_normalize(scaled.astype(np.float32))


def select_farthest_anchors(
    embeddings: np.ndarray,
    count: int,
    seed: int = 7,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    raw = l2_normalize(embeddings.astype(np.float32))
    candidate_ids = np.arange(len(raw), dtype=np.int32) if candidates is None else candidates.astype(np.int32)
    rng = np.random.default_rng(seed)
    first = int(candidate_ids[int(rng.integers(0, len(candidate_ids)))])
    selected = [first]
    max_sim = raw[candidate_ids] @ raw[first]
    for _ in range(1, min(count, len(candidate_ids))):
        next_idx = int(candidate_ids[int(np.argmin(max_sim))])
        selected.append(next_idx)
        max_sim = np.maximum(max_sim, raw[candidate_ids] @ raw[next_idx])
    return np.array(selected, dtype=np.int32)


@dataclass
class DriftStats:
    doc_count: int
    mean_anchor_similarity: float
    mean_top_anchor_similarity: float
    mean_signature_entropy: float
    anchor_usage_gini: float


@dataclass
class MemoryEstimate:
    doc_count: int
    raw_dim: int
    signature_dim: int
    raw_float32_mb: float
    signature_float32_mb: float
    signature_float16_mb: float
    signature_int8_mb: float


class LARPHNSWIndex:
    """HNSW-backed fixed-anchor relative routing index with raw embedding rerank."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        anchor_count: int = 256,
        signature_transform: str = "row_zscore",
        seed: int = 7,
        space: str = "cosine",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
        max_elements: int | None = None,
        anchor_weights: np.ndarray | None = None,
    ):
        self.model_name = model_name
        self.anchor_count = anchor_count
        self.signature_transform = signature_transform
        self.seed = seed
        self.space = space
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.anchor_weights = None if anchor_weights is None else np.asarray(anchor_weights, dtype=np.float32)

        self.doc_ids: list[str] = []
        self.raw_embeddings: np.ndarray | None = None
        self.relative_signatures: np.ndarray | None = None
        self.anchor_indices: np.ndarray | None = None
        self.anchor_embeddings: np.ndarray | None = None
        self.relative_hnsw: hnswlib.Index | None = None
        self.raw_hnsw: hnswlib.Index | None = None

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def _transform(self, sig: np.ndarray) -> np.ndarray:
        x = sig.astype(np.float32)
        if self.anchor_weights is not None:
            x = x * self.anchor_weights.reshape(1, -1)
        if self.signature_transform == "raw":
            return l2_normalize(x)
        if self.signature_transform == "row_zscore":
            return row_zscore(x)
        raise ValueError(f"Unsupported signature transform: {self.signature_transform}")

    def _signature_from_raw(self, embeddings: np.ndarray) -> np.ndarray:
        if self.anchor_embeddings is None:
            raise RuntimeError("Index has no anchor embeddings.")
        raw = l2_normalize(embeddings.astype(np.float32))
        return self._transform(raw @ self.anchor_embeddings.T)

    def _make_hnsw(self, dim: int, max_elements: int) -> hnswlib.Index:
        index = hnswlib.Index(space=self.space, dim=dim)
        index.init_index(max_elements=max_elements, ef_construction=self.ef_construction, M=self.hnsw_m)
        index.set_ef(self.ef_search)
        return index

    def _ensure_capacity(self, extra: int) -> None:
        target = self.size + extra
        if self.relative_hnsw is None or self.raw_hnsw is None:
            return
        current = self.relative_hnsw.get_max_elements()
        if target <= current:
            return
        new_cap = max(target, int(current * 1.5) + 1)
        self.relative_hnsw.resize_index(new_cap)
        self.raw_hnsw.resize_index(new_cap)
        self.max_elements = new_cap

    def fit_embeddings(
        self,
        embeddings: np.ndarray,
        doc_ids: Iterable[str] | None = None,
        anchor_indices: np.ndarray | None = None,
    ) -> "LARPHNSWIndex":
        raw = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self.doc_ids = list(doc_ids) if doc_ids is not None else [str(i) for i in range(len(raw))]
        if len(self.doc_ids) != len(raw):
            raise ValueError("doc_ids length must match embeddings length.")
        if anchor_indices is None:
            anchor_indices = select_farthest_anchors(raw, self.anchor_count, self.seed)
        self.anchor_indices = np.asarray(anchor_indices, dtype=np.int32)
        self.anchor_count = len(self.anchor_indices)
        self.anchor_embeddings = raw[self.anchor_indices]
        if self.anchor_weights is not None and len(self.anchor_weights) != self.anchor_count:
            raise ValueError("anchor_weights length must match anchor_count.")
        self.raw_embeddings = raw
        self.relative_signatures = self._signature_from_raw(raw)

        max_elements = self.max_elements or max(len(raw) * 2, len(raw) + 1)
        self.relative_hnsw = self._make_hnsw(self.relative_signatures.shape[1], max_elements)
        self.raw_hnsw = self._make_hnsw(self.raw_embeddings.shape[1], max_elements)
        labels = np.arange(len(raw), dtype=np.int64)
        self.relative_hnsw.add_items(self.relative_signatures, labels)
        self.raw_hnsw.add_items(self.raw_embeddings, labels)
        return self

    def fit_texts(self, texts: list[str], doc_ids: Iterable[str] | None = None, batch_size: int = 32) -> "LARPHNSWIndex":
        embedder = HFMeanPoolEmbedder(self.model_name)
        return self.fit_embeddings(embedder.encode(texts, batch_size=batch_size), doc_ids=doc_ids)

    def insert_embeddings(self, embeddings: np.ndarray, doc_ids: Iterable[str]) -> None:
        if self.raw_embeddings is None or self.relative_signatures is None:
            raise RuntimeError("Call fit_embeddings before inserting.")
        raw = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        ids = [str(doc_id) for doc_id in doc_ids]
        if len(raw) != len(ids):
            raise ValueError("doc_ids length must match embeddings length.")
        start = self.size
        labels = np.arange(start, start + len(raw), dtype=np.int64)
        sig = self._signature_from_raw(raw)
        self._ensure_capacity(len(raw))
        self.relative_hnsw.add_items(sig, labels)
        self.raw_hnsw.add_items(raw, labels)
        self.raw_embeddings = np.vstack([self.raw_embeddings, raw])
        self.relative_signatures = np.vstack([self.relative_signatures, sig])
        self.doc_ids.extend(ids)

    def insert_embedding(self, embedding: np.ndarray, doc_id: str) -> None:
        self.insert_embeddings(np.asarray(embedding, dtype=np.float32).reshape(1, -1), [doc_id])

    def add_texts(self, texts: list[str], doc_ids: Iterable[str], batch_size: int = 32) -> None:
        embedder = HFMeanPoolEmbedder(self.model_name)
        self.insert_embeddings(embedder.encode(texts, batch_size=batch_size), doc_ids)

    def candidate_indices(self, query_embedding: np.ndarray, pool: int = 250) -> tuple[np.ndarray, np.ndarray]:
        if self.relative_hnsw is None:
            raise RuntimeError("Index is empty.")
        query_raw = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        query_sig = self._signature_from_raw(query_raw)
        labels, distances = self.relative_hnsw.knn_query(query_sig, k=min(pool, self.size))
        rel_scores = 1.0 - distances[0]
        return labels[0].astype(np.int64), rel_scores.astype(np.float32)

    def search_embedding(self, query_embedding: np.ndarray, top_k: int = 10, pool: int = 250) -> list[SearchResult]:
        if self.raw_embeddings is None:
            raise RuntimeError("Index is empty.")
        query_raw = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
        candidates, rel_scores = self.candidate_indices(query_raw, pool=pool)
        raw_scores = self.raw_embeddings[candidates] @ query_raw
        order = top_indices(raw_scores, min(top_k, len(candidates)))
        return [
            SearchResult(
                doc_id=self.doc_ids[int(candidates[pos])],
                score=float(raw_scores[pos]),
                relative_score=float(rel_scores[pos]),
                index=int(candidates[pos]),
            )
            for pos in order
        ]

    def search_relative_only(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        candidates, rel_scores = self.candidate_indices(query_embedding, pool=top_k)
        return [
            SearchResult(
                doc_id=self.doc_ids[int(idx)],
                score=float(rel_score),
                relative_score=float(rel_score),
                index=int(idx),
            )
            for idx, rel_score in zip(candidates, rel_scores)
        ]

    def search_raw_hnsw(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        if self.raw_hnsw is None or self.raw_embeddings is None:
            raise RuntimeError("Index is empty.")
        query_raw = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        labels, distances = self.raw_hnsw.knn_query(query_raw, k=min(top_k, self.size))
        scores = 1.0 - distances[0]
        return [
            SearchResult(
                doc_id=self.doc_ids[int(idx)],
                score=float(score),
                relative_score=float(score),
                index=int(idx),
            )
            for idx, score in zip(labels[0], scores)
        ]

    def search_text(self, query: str, top_k: int = 10, pool: int = 250, embedder: HFMeanPoolEmbedder | None = None) -> list[SearchResult]:
        embedder = embedder or HFMeanPoolEmbedder(self.model_name)
        return self.search_embedding(embedder.encode([query])[0], top_k=top_k, pool=pool)

    def drift_stats(self, embeddings: np.ndarray) -> DriftStats:
        sig_raw = l2_normalize(np.asarray(embeddings, dtype=np.float32)) @ self.anchor_embeddings.T
        top_sim = np.max(sig_raw, axis=1)
        probs = np.exp(sig_raw - sig_raw.max(axis=1, keepdims=True))
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
        top_anchor = np.argmax(sig_raw, axis=1)
        counts = np.bincount(top_anchor, minlength=self.anchor_count).astype(np.float64)
        sorted_counts = np.sort(counts)
        if sorted_counts.sum() == 0:
            gini = 0.0
        else:
            n = len(sorted_counts)
            gini = float((2 * np.arange(1, n + 1) - n - 1) @ sorted_counts / (n * sorted_counts.sum()))
        return DriftStats(
            doc_count=len(sig_raw),
            mean_anchor_similarity=float(sig_raw.mean()),
            mean_top_anchor_similarity=float(top_sim.mean()),
            mean_signature_entropy=float(entropy.mean()),
            anchor_usage_gini=gini,
        )

    def memory_estimate(self) -> MemoryEstimate:
        if self.raw_embeddings is None or self.relative_signatures is None:
            raise RuntimeError("Index is empty.")
        n, raw_dim = self.raw_embeddings.shape
        sig_dim = self.relative_signatures.shape[1]
        return MemoryEstimate(
            doc_count=n,
            raw_dim=raw_dim,
            signature_dim=sig_dim,
            raw_float32_mb=n * raw_dim * 4 / 1_000_000,
            signature_float32_mb=n * sig_dim * 4 / 1_000_000,
            signature_float16_mb=n * sig_dim * 2 / 1_000_000,
            signature_int8_mb=n * sig_dim / 1_000_000,
        )

    def benchmark_queries(self, query_embeddings: np.ndarray, top_k: int = 10, pool: int = 250) -> dict[str, float]:
        start = time.perf_counter()
        for emb in query_embeddings:
            self.search_embedding(emb, top_k=top_k, pool=pool)
        elapsed = time.perf_counter() - start
        return {"query_count": len(query_embeddings), "total_seconds": elapsed, "ms_per_query": 1000 * elapsed / max(1, len(query_embeddings))}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_name": self.model_name,
            "anchor_count": self.anchor_count,
            "signature_transform": self.signature_transform,
            "seed": self.seed,
            "space": self.space,
            "hnsw_m": self.hnsw_m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_elements": self.max_elements,
            "doc_ids": self.doc_ids,
            "has_anchor_weights": self.anchor_weights is not None,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.save(path / "raw_embeddings.npy", self.raw_embeddings)
        np.save(path / "relative_signatures.npy", self.relative_signatures)
        np.save(path / "anchor_indices.npy", self.anchor_indices)
        np.save(path / "anchor_embeddings.npy", self.anchor_embeddings)
        if self.anchor_weights is not None:
            np.save(path / "anchor_weights.npy", self.anchor_weights)
        self.relative_hnsw.save_index(str(path / "relative_hnsw.bin"))
        self.raw_hnsw.save_index(str(path / "raw_hnsw.bin"))

    @classmethod
    def load(cls, path: str | Path) -> "LARPHNSWIndex":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        anchor_weights = np.load(path / "anchor_weights.npy") if metadata.get("has_anchor_weights") else None
        index = cls(
            model_name=metadata["model_name"],
            anchor_count=metadata["anchor_count"],
            signature_transform=metadata["signature_transform"],
            seed=metadata["seed"],
            space=metadata["space"],
            hnsw_m=metadata["hnsw_m"],
            ef_construction=metadata["ef_construction"],
            ef_search=metadata["ef_search"],
            max_elements=metadata["max_elements"],
            anchor_weights=anchor_weights,
        )
        index.doc_ids = list(metadata["doc_ids"])
        index.raw_embeddings = np.load(path / "raw_embeddings.npy")
        index.relative_signatures = np.load(path / "relative_signatures.npy")
        index.anchor_indices = np.load(path / "anchor_indices.npy")
        index.anchor_embeddings = np.load(path / "anchor_embeddings.npy")
        index.relative_hnsw = hnswlib.Index(space=index.space, dim=index.relative_signatures.shape[1])
        index.relative_hnsw.load_index(str(path / "relative_hnsw.bin"), max_elements=index.max_elements or len(index.doc_ids))
        index.relative_hnsw.set_ef(index.ef_search)
        index.raw_hnsw = hnswlib.Index(space=index.space, dim=index.raw_embeddings.shape[1])
        index.raw_hnsw.load_index(str(path / "raw_hnsw.bin"), max_elements=index.max_elements or len(index.doc_ids))
        index.raw_hnsw.set_ef(index.ef_search)
        return index


class MultiGenerationLARP:
    """Search facade for active anchor generations during lazy migration."""

    def __init__(self, generations: list[LARPHNSWIndex] | None = None):
        self.generations = generations or []

    def add_generation(self, index: LARPHNSWIndex) -> None:
        self.generations.append(index)

    def search_embedding(self, query_embedding: np.ndarray, top_k: int = 10, pool_per_generation: int = 250) -> list[SearchResult]:
        results = []
        for generation in self.generations:
            results.extend(generation.search_embedding(query_embedding, top_k=top_k, pool=pool_per_generation))
        if not results:
            return []
        results.sort(key=lambda r: r.score, reverse=True)
        seen = set()
        merged = []
        for result in results:
            if result.doc_id in seen:
                continue
            seen.add(result.doc_id)
            merged.append(result)
            if len(merged) >= top_k:
                break
        return merged
