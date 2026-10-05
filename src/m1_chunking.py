from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    import numpy as np

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # normalize → dot product chính là cosine similarity
    embeddings = _get_semantic_model().encode(sentences, normalize_embeddings=True)
    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if float(np.dot(embeddings[i - 1], embeddings[i])) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [Chunk(text=" ".join(g), metadata={**metadata, "strategy": "semantic", "chunk_index": i})
            for i, g in enumerate(groups)]


_semantic_model = None


def _get_semantic_model():
    """Load all-MiniLM-L6-v2 một lần, dùng lại cho mọi document."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    # Prefix theo source để parent_id không trùng giữa các document
    prefix = f"{metadata['source']}::" if metadata.get("source") else ""
    parents, children = [], []
    for parent_text in _split_to_size(text, parent_size):
        pid = f"{prefix}parent_{len(parents)}"
        parents.append(Chunk(text=parent_text,
                             metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
        for child_text in _split_to_size(parent_text, child_size):
            children.append(Chunk(text=child_text,
                                  metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                                  parent_id=pid))
    return (parents, children)


# Ranh giới cắt từ lớn → nhỏ: paragraph → dòng → câu → từ
_SPLIT_LEVELS = [(r"\n\n", "\n\n"), (r"\n", "\n"), (r"(?<=[.!?;])\s+", " "), (r"\s+", " ")]


def _split_to_size(text: str, max_size: int, level: int = 0) -> list[str]:
    """Chia text thành các đoạn ≤ max_size chars.

    Cắt tại ranh giới lớn nhất có thể, rồi gộp tham lam các mảnh liền kề
    để mỗi đoạn càng gần max_size càng tốt (1 từ dài hơn max_size thì giữ nguyên).
    """
    text = text.strip()
    if len(text) <= max_size or level >= len(_SPLIT_LEVELS):
        return [text] if text else []

    pattern, joiner = _SPLIT_LEVELS[level]
    pieces = [p for part in re.split(pattern, text) for p in _split_to_size(part, max_size, level + 1)]

    merged, current = [], ""
    for piece in pieces:
        if current and len(current) + len(joiner) + len(piece) > max_size:
            merged.append(current)
            current = piece
        else:
            current = f"{current}{joiner}{piece}" if current else piece
    if current:
        merged.append(current)
    return merged


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    # re.split với capture group → [content_trước_header_đầu, header1, content1, header2, content2, ...]
    parts = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    chunks = []

    if parts[0].strip():
        chunks.append(Chunk(text=parts[0].strip(), metadata={
            **metadata, "section": "", "section_path": "", "strategy": "structure", "chunk_index": 0}))

    titles: dict[int, str] = {}  # level → title của header gần nhất, để dựng section_path
    for header, content in zip(parts[1::2], parts[2::2]):
        hashes, title = header.strip().split(maxsplit=1)
        level = len(hashes)
        titles = {lv: t for lv, t in titles.items() if lv < level}
        titles[level] = title.strip()
        if not content.strip():  # header không có nội dung riêng (vd. "# Nghỉ phép" ngay trước "## ...")
            continue
        chunks.append(Chunk(text=f"{header.strip()}\n{content.strip()}", metadata={
            **metadata,
            "section": title.strip(),
            "section_path": " > ".join(titles[lv] for lv in sorted(titles)),
            "strategy": "structure",
            "chunk_index": len(chunks),
        }))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
