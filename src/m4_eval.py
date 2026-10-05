from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

_ragas_models = None


def _get_ragas_models():
    """LLM judge + embeddings cho RAGAS (theo LLM_PROVIDER), load 1 lần/process."""
    global _ragas_models
    if _ragas_models is None:
        from config import get_ragas_models
        _ragas_models = get_ragas_models()
    return _ragas_models


def _safe(x) -> float:
    """RAGAS trả NaN khi judge lỗi/parse fail → đổi thành 0.0 để JSON hợp lệ và sort được."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(x) else x


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {m: 0.0 for m in METRICS}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        import openai
        from ragas.run_config import RunConfig

        llm, embeddings = _get_ragas_models()
        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # Mặc định RAGAS retry MỌI exception 10 lần (backoff ≤60s) → lỗi vĩnh viễn như 401/402
        # (sai key / hết credit) treo vài phút. Chỉ retry lỗi tạm thời; lỗi khác → NaN → 0.0.
        run_config = RunConfig(timeout=120, max_retries=5, max_wait=30, exception_types=(
            openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError))
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                          llm=llm, embeddings=embeddings, run_config=run_config)
        df = result.to_pandas()
        per_question = [EvalResult(question=row["question"], answer=row["answer"],
                                   contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                                   **{m: _safe(row.get(m)) for m in METRICS})
                        for _, row in df.iterrows()]
        # Aggregate = nanmean của RAGAS (bỏ qua câu judge lỗi), giống cách RAGAS tự báo cáo
        return {**{m: round(_safe(df[m].mean()), 4) for m in METRICS}, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {**zeros, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Diagnostic Tree: metric thấp nhất → tầng lỗi (generation / retrieval / ranking) → cách sửa
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — câu trả lời có claim không có trong context",
                         "Tighten prompt (chỉ dùng context), temperature=0, bắt trích dẫn nguồn"),
        "context_recall": ("Missing relevant chunks — context thiếu thông tin cần cho ground truth",
                           "Improve chunking (parent-child / giữ nguyên bảng), tăng top_k, thêm BM25/HyQA"),
        "context_precision": ("Too many irrelevant chunks — chunk liên quan bị xếp dưới chunk nhiễu",
                              "Add reranking, metadata filter (version/hiệu lực), giảm top_k"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lan man hoặc né câu hỏi",
                             "Improve prompt template: trả lời thẳng vào câu hỏi, ngắn gọn"),
    }

    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "scores": scores,
            "avg_score": round(sum(scores.values()) / len(scores), 4),
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return sorted(analyzed, key=lambda x: x["avg_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
