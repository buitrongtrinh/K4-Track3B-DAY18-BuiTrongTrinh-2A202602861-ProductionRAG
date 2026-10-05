from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import json, os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# parent_id → parent text: retrieve child (precision) → trả về parent (đủ context, không cắt bảng)
_PARENTS: dict[str, str] = {}
# Latency breakdown (ms): thời gian build từng bước + từng bước của mỗi query
_BUILD_TIMES: dict[str, float] = {}
_QUERY_TIMES: list[dict[str, float]] = []

ANSWER_PROMPT = """Bạn là trợ lý tra cứu chính sách nội bộ công ty. Trả lời câu hỏi CHỈ dựa trên context.
Quy tắc:
- Trả lời thẳng vào câu hỏi bằng tiếng Việt, ngắn gọn (1-3 câu), nêu rõ con số và điều kiện liên quan.
- Nếu context có nhiều phiên bản của cùng một chính sách: dùng phiên bản hiện hành (ngày hiệu lực mới nhất), \
có thể nhắc phiên bản cũ đã bị thay thế.
- Câu hỏi cần tính toán: áp dụng đúng quy định trong context và nêu phép tính.
- Không thêm thông tin ngoài context. Nếu context không có thông tin cần thiết → trả lời "Không tìm thấy thông tin."
"""


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        _PARENTS.update({p.metadata["parent_id"]: p.text for p in parents})
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    _BUILD_TIMES["chunking"] = (time.time() - t0) * 1000
    print(f"  ✓ {len(all_chunks)} child chunks ({len(_PARENTS)} parents) from {len(docs)} documents "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    _BUILD_TIMES["enrichment"] = (time.time() - t0) * 1000

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    _BUILD_TIMES["indexing"] = (time.time() - t0) * 1000
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()  # load ngay để latency query đầu tiên không tính thời gian load model
    _BUILD_TIMES["reranker_load"] = (time.time() - t0) * 1000
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    timings = {}
    t0 = time.perf_counter()
    results = search.search(query)
    timings["hybrid_search"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    # Rerank toàn bộ children, rồi lấy top-k parent KHÁC NHAU theo thứ tự rerank
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    ranked = [(r.text, r.metadata) for r in (reranked or results)]
    contexts, seen = [], set()
    for text, metadata in ranked:
        pid = metadata.get("parent_id")
        if (pid or text) in seen:
            continue
        seen.add(pid or text)
        contexts.append(_PARENTS.get(pid, text))
        if len(contexts) == RERANK_TOP_K:
            break
    timings["rerank"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    from config import LLM_API_KEY, LLM_MODEL, get_llm_client
    if LLM_API_KEY and contexts:
        try:
            client = get_llm_client()
            context_str = "\n\n---\n\n".join(contexts)
            resp = client.chat.completions.create(model=LLM_MODEL, temperature=0, messages=[
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["generation"] = (time.perf_counter() - t0) * 1000
    _QUERY_TIMES.append(timings)
    return answer, contexts


def report_latency(path: str = "reports/latency_report.json") -> dict:
    """In + lưu bảng latency từng bước (build 1 lần + trung bình/p95 mỗi query)."""
    steps = ["hybrid_search", "rerank", "generation"]
    per_query = {}
    for step in steps:
        values = sorted(t[step] for t in _QUERY_TIMES)
        if values:
            per_query[step] = {"avg_ms": round(sum(values) / len(values), 1),
                               "p95_ms": round(values[min(len(values) - 1, int(0.95 * len(values)))], 1),
                               "max_ms": round(values[-1], 1)}
    totals = sorted(sum(t[s] for s in steps) for t in _QUERY_TIMES)
    if totals:
        per_query["total"] = {"avg_ms": round(sum(totals) / len(totals), 1),
                              "p95_ms": round(totals[min(len(totals) - 1, int(0.95 * len(totals)))], 1),
                              "max_ms": round(totals[-1], 1)}

    print("\nLATENCY BREAKDOWN")
    print(f"  {'Build step':<16} {'ms':>10}")
    for step, ms in _BUILD_TIMES.items():
        print(f"  {step:<16} {ms:>10.0f}")
    print(f"\n  {'Query step':<16} {'avg ms':>10} {'p95 ms':>10} {'max ms':>10}")
    for step, st in per_query.items():
        print(f"  {step:<16} {st['avg_ms']:>10.1f} {st['p95_ms']:>10.1f} {st['max_ms']:>10.1f}")

    report = {"build_ms": {k: round(v, 1) for k, v in _BUILD_TIMES.items()},
              "per_query_ms": per_query, "num_queries": len(_QUERY_TIMES)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    report_latency()
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
