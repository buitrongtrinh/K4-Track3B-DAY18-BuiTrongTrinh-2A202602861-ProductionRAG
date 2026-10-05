from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")
    # underthesea nối từ ghép bằng "_" ("nghỉ_phép") → query "nghỉ phép" sẽ không khớp token
    return segmented.replace("_", " ")


def _bm25_tokenize(text: str) -> list[str]:
    """Segment + lowercase + bỏ token chỉ gồm dấu câu ("**", "|", "." ...)."""
    return [t for t in segment_vietnamese(text).lower().split() if any(ch.isalnum() for ch in t)]


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [_bm25_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if chunks else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_bm25_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [SearchResult(text=self.documents[i]["text"], score=float(scores[i]),
                             metadata=self.documents[i].get("metadata", {}), method="bm25")
                for i in top_indices if scores[i] > 0]


_encoder = None


def get_encoder():
    """bge-m3 dùng chung trong process (baseline, production, RAGAS) — tránh load nhiều bản lên GPU."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(EMBEDDING_MODEL)
        if _encoder.device.type == "cuda":
            _encoder.half()  # fp16: ~1.1GB VRAM thay vì ~2.2GB
    return _encoder


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = get_encoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # recreate_collection() đã deprecated → xoá rồi tạo lại
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=True)
        points = [PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": c["text"]})
                  for i, (c, v) in enumerate(zip(chunks, vectors))]
        for start in range(0, len(points), 256):
            self.client.upsert(collection, points[start:start + 256])

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(collection, query=query_vector, limit=top_k)
        return [SearchResult(text=pt.payload["text"], score=pt.score, metadata=pt.payload, method="dense")
                for pt in response.points]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}  # text → {"score": float, "result": SearchResult}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)[:top_k]
    return [SearchResult(text=e["result"].text, score=e["score"], metadata=e["result"].metadata, method="hybrid")
            for e in ranked]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
