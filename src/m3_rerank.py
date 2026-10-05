from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_MODEL_CACHE: dict = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Dùng sentence_transformers.CrossEncoder (FlagReranker crash với transformers>=5.0)
            if self.model_name not in _MODEL_CACHE:  # load 1 lần/process, dùng chung giữa các instance
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(self.model_name)
                if model.device.type == "cuda":
                    model.half()  # fp16: ~1.1GB VRAM thay vì ~2.2GB, điểm rerank gần như không đổi
                _MODEL_CACHE[self.model_name] = model
            self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        scores = model.predict([(query, doc["text"]) for doc in documents])
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
        return [RerankResult(text=doc["text"], original_score=doc.get("score", 0.0),
                             rerank_score=float(score), metadata=doc.get("metadata", {}), rank=i)
                for i, (score, doc) in enumerate(scored[:top_k])]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        from flashrank import Ranker, RerankRequest

        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        return [RerankResult(text=r["text"], original_score=documents[r["id"]].get("score", 0.0),
                             rerank_score=float(r["score"]), metadata=documents[r["id"]].get("metadata", {}),
                             rank=i)
                for i, r in enumerate(results[:top_k])]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
