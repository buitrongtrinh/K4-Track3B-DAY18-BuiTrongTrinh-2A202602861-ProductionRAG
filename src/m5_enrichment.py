from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import hashlib, json, os, re, sys, threading
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, get_llm_client


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── LLM helper (provider theo config: openai | deepseek) ─

_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "llm_cache.json")
_cache: dict | None = None
_cache_lock = threading.Lock()


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi LLM (cache theo prompt → chạy lại pipeline không tốn thêm API call)."""
    key = hashlib.sha256(json.dumps([LLM_MODEL, system, user, max_tokens, json_mode]).encode()).hexdigest()
    with _cache_lock:
        if key in _load_cache():
            return _cache[key]

    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0, **extra,
    )
    content = resp.choices[0].message.content.strip()

    with _cache_lock:
        _load_cache()[key] = content
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
    return content


_documents: dict[str, str] | None = None


def _document_text(source: str) -> str:
    """Toàn văn document theo tên file — để LLM biết chunk nằm ở đâu (Anthropic contextual retrieval)."""
    global _documents
    if _documents is None:
        from src.m1_chunking import load_documents
        _documents = {d["metadata"]["source"]: d["text"] for d in load_documents()}
    return _documents.get(source, "")


def _context_prompt(text: str, source: str) -> str:
    doc = _document_text(source)
    doc_block = f"<document name=\"{source}\">\n{doc[:6000]}\n</document>\n\n" if doc else f"Tài liệu: {source}\n\n"
    return f"{doc_block}<chunk>\n{text}\n</chunk>"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if LLM_API_KEY:
        try:
            return _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Giữ nguyên các con số.",
                         text, max_tokens=150)
        except Exception as e:
            print(f"  ⚠️  LLM summarize failed: {e}")

    # Extractive fallback (không cần API): 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if LLM_API_KEY:
        try:
            content = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời, "
                            "theo cách một nhân viên sẽ hỏi. Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
                            text, max_tokens=200)
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in content.split("\n")]
            return [q for q in questions if q][:n_questions]
        except Exception as e:
            print(f"  ⚠️  LLM HyQA failed: {e}")

    # Extractive fallback: biến các câu dài thành dạng câu hỏi
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if LLM_API_KEY:
        try:
            context = _chat("Viết 1 câu ngắn bằng tiếng Việt đặt đoạn trích (chunk) vào ngữ cảnh của tài liệu: "
                            "tên tài liệu/chính sách, phiên bản và ngày hiệu lực (nếu có), mục nào, nói về gì. "
                            "Chỉ trả về đúng 1 câu.",
                            _context_prompt(text, document_title), max_tokens=120)
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  LLM contextual failed: {e}")

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────

_METADATA_KEYS = {"topic", "entities", "category", "language", "version", "effective_date"}
_METADATA_SCHEMA = ('{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|safety", '
                    '"language": "vi|en", "version": "... hoặc null", "effective_date": "dd/mm/yyyy hoặc null"}')


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if LLM_API_KEY:
        try:
            meta = json.loads(_chat(f"Trích xuất metadata từ đoạn văn. Trả về JSON đúng schema: {_METADATA_SCHEMA}",
                                    text, max_tokens=200, json_mode=True))
            return {k: v for k, v in meta.items() if k in _METADATA_KEYS}
        except Exception as e:
            print(f"  ⚠️  LLM metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if not LLM_API_KEY:
        return {}
    try:
        result = json.loads(_chat(
            f"""Bạn chuẩn bị dữ liệu cho hệ thống tìm kiếm tài liệu nội bộ. Đọc toàn bộ tài liệu và đoạn trích (chunk), trả về JSON:
{{
  "summary": "tóm tắt chunk trong 2-3 câu, giữ nguyên con số",
  "questions": ["3 câu hỏi tiếng Việt mà chunk trả lời được, theo cách nhân viên hỏi"],
  "context": "1 câu đặt chunk vào ngữ cảnh tài liệu: tên chính sách, phiên bản, ngày hiệu lực, mục nào, nói về gì",
  "metadata": {_METADATA_SCHEMA}
}}""",
            _context_prompt(text, source), max_tokens=600, json_mode=True))
        result["metadata"] = {k: v for k, v in result.get("metadata", {}).items() if k in _METADATA_KEYS}
        return result
    except Exception as e:
        print(f"  ⚠️  Enrichment API failed: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    def _enrich_one(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )

    # LLM calls là I/O-bound → chạy song song, map() giữ nguyên thứ tự chunks
    from concurrent.futures import ThreadPoolExecutor

    enriched = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, item in enumerate(pool.map(_enrich_one, chunks)):
            enriched.append(item)
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
