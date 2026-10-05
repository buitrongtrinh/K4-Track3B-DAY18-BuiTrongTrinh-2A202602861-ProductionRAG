# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Bùi Trọng Trinh (2A202602861)
**Khóa:** K4 - Track 3B
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo **208 chunks** (avg 99 ký tự, min 6) vs basic **51 chunks** (avg 410). `all-MiniLM-L6-v2` là model tiếng Anh nên similarity giữa các câu tiếng Việt thấp → cắt quá vụn; với corpus tiếng Việt cần model đa ngữ hoặc threshold thấp hơn. |
| Hierarchical (parent-child) | M1 + pipeline | `chunk_hierarchical()`, `run_query()` | 26 parents / 119 children (≤256 ký tự). Child 256 ký tự **cắt đôi bảng phê duyệt** trong `mua_sam.md`, nhưng vì pipeline rerank trên child rồi trả về **parent** nên LLM vẫn thấy đủ bảng → câu "55 triệu → CEO" trả lời đúng. Context Recall 0.85 → 0.925. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF (k=60) gộp 2 bảng xếp hạng không cần chuẩn hoá điểm. Ví dụ: câu "nghỉ bao nhiêu ngày phép năm" BM25 top-1 sai (`nghi_phep_dac_biet`), dense đúng → hybrid top-1 đúng; câu MFA BM25 kéo `vpn_truy_cap` vào top-3, hybrid loại ra. 20/20 câu có tài liệu đúng ở hybrid top-1. Phải `replace("_", " ")` sau underthesea và bỏ token dấu câu (`**`, `\|` của bảng markdown). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 fp16: **~105 ms** cho 20 children (p95 125 ms), model load 5 s. Context Precision 0.75 → **0.875**. Điểm rerank tách rõ (0.99 vs 0.00) nhưng không phân biệt được 2 version gần giống nhau (v2023/v2024 đều 0.99 → failure #4). Flashrank chỉ ~3 ms nhưng model TinyBERT tiếng Anh. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: F 0.8975 / AR 0.8443 / CP 0.875 / CR 0.925. **Answer Relevancy thấp nhất** vì câu multi-hop #1 trả "Không tìm thấy" một phần → RAGAS gắn cờ noncommittal → 0 cho câu đó. Chỉ answer_relevancy cần embeddings (bge-m3); 3 metric còn lại chỉ dùng LLM judge. |
| Contextual embeddings | M5 | `_enrich_single_call()` | 1 call/chunk (JSON mode) trả summary + 3 HyQA + 1 câu context + metadata (`version`, `effective_date`). LLM được xem **toàn văn document** (kiểu Anthropic contextual retrieval) nên câu context nêu được "Chính sách nghỉ phép năm phiên bản 2024" cho cả các child không chứa dòng phiên bản. 119 chunks / 8 luồng ≈ 27 s, có cache theo prompt. Chưa ablation riêng tác động của M5. |
| Latency breakdown | pipeline | `report_latency()` | Mỗi query ~1 061 ms: search 16 ms, rerank 105 ms, **generation 940 ms (~89%)** → tối ưu latency phải nhắm vào LLM (streaming, model nhỏ hơn, cache), không phải retrieval. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. DeepSeek không hỗ trợ `n > 1`**
- **Exact error:** `openai.BadRequestError: Error code: 400 - {'error': {'message': 'Invalid n value (currently only n = 1 is supported)', 'type': 'invalid_request_error'}}`
- **Debug:** Gọi thử API trực tiếp với `n=3` → tái hiện lỗi. Đọc source `ragas.llms.base.LangchainLLMWrapper.generate_text`: nếu LLM là `ChatOpenAI` (nằm trong `MULTIPLE_COMPLETION_SUPPORTED`) thì gửi `n` trong 1 request; ngược lại gửi n request riêng. `answer_relevancy` dùng `strictness=3`.
- **Giải quyết:** Bọc `ChatOpenAI` trong `SingleCompletionChat(BaseChatModel)` (`config._single_completion`) → RAGAS tự fallback sang n request với n=1. Test lại: answer_relevancy 0.92, faithfulness 1.0 trên mẫu thử.

**2. CUDA out of memory trên GPU 6GB**
- **Exact error:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 5.72 GiB of which 19.69 MiB is free.` (khi load reranker trong `build_pipeline()`)
- **Debug:** Đo `torch.cuda.memory_allocated()` từng bước theo đúng thứ tự `main.py`: bge-m3 fp16 1.06 GiB → + `HuggingFaceEmbeddings` bge-m3 fp32 cho RAGAS **3.17 GiB** → + reranker (load fp32 rồi mới `.half()`) → OOM tại 5.11 GiB. `nvidia-smi --query-compute-apps` cho thấy thêm 1 process khác giữ ~0.46 GiB (và một container GPU khác giữ 2 GiB ở lần đầu).
- **Giải quyết:** (a) bge-m3 và reranker load **1 lần/process** và chạy fp16; (b) RAGAS dùng chung instance bge-m3 qua adapter LangChain `Embeddings` thay vì load bản fp32 thứ 2; (c) dừng container GPU không liên quan khi chạy lab.

**3. Test chậm có nguy cơ vượt timeout của `check_lab.py` (120 s)**
- **Hiện tượng:** `pytest tests/test_m3.py` mất 29.8 s vì mỗi test tạo `CrossEncoderReranker()` mới → load model 5 lần.
- **Giải quyết:** cache model ở module level → M2 + M3 còn 9.2 s; M5 cache LLM response theo hash prompt.

**4. PDF scan không có text layer**
- **Hiện tượng:** `⚠️ Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR).` (cả Nghị định 13/2023).
- **Đánh giá:** Test set không hỏi 2 file này nên không ảnh hưởng điểm, nhưng production thật cần bước OCR (vd. PaddleOCR / Tesseract `vie`) trước khi chunk.

**5. Test RAGAS treo hơn 2 phút khi API hết credit**
- **Exact error:** `HTTP/1.1 402 Payment Required` (log httpx, lặp lại 9+ lần trong ~60 s) → `check_lab.py`: `timed out after 120 seconds`.
- **Debug:** `pytest --durations` chỉ ra `test_evaluate_returns_metrics` mất 127.9 s; bật log httpx thấy RAGAS retry liên tục. Nguyên nhân: `RunConfig` mặc định retry **mọi** exception 10 lần với backoff tới 60 s, kể cả lỗi vĩnh viễn (401/402).
- **Giải quyết:** `RunConfig(exception_types=(RateLimitError, APITimeoutError, APIConnectionError, InternalServerError), max_retries=5, max_wait=30)` → lỗi vĩnh viễn fail ngay (NaN → 0.0), test còn 4 s.

**Kiến thức còn thiếu & cách bổ sung:**
- Cơ chế bên trong RAGAS (metric nào cần embeddings, n-completions, noncommittal flag) → đọc trực tiếp source trong `site-packages/ragas/metrics/` thay vì chỉ đọc docs.
- Ước lượng VRAM cho nhiều model cùng process (fp32 vs fp16, peak khi load) → đo bằng `torch.cuda.memory_allocated()` từng bước thay vì đoán.

---

## Phần 3: Kế hoạch hành động (Action Plan)

## Project: TubeNote — RAG hỏi đáp trên video YouTube đã lồng tiếng Việt

### Hiện tại
- **RAG pipeline:** subtitle (`text_vi` → `text_tts` → `text`) → chunk theo segment có timestamp (900–1 200 ký tự, overlap 2 segment) → Chroma + `BAAI/bge-m3` dense + BM25 sparse → RRF → LLM (mặc định DeepSeek), có cached video summary mở đầu phiên chat.
- **Known issues:**
  - Chưa có reranker → top-k sau RRF còn nhiễu (giống Context Precision 0.75 của baseline lab).
  - Chunk lớn (≤1 200 ký tự) vừa dùng để retrieve vừa dùng để trả lời → precision retrieval thấp với câu hỏi chi tiết.
  - Chưa có bộ đánh giá → không đo được thay đổi nào làm tốt/xấu đi.
  - Chạy chung GPU với ASR (whisper) + TTS (OmniVoice) → rủi ro OOM như đã gặp trong lab.

### Plan áp dụng
1. [ ] **Evaluation trước tiên:** tạo test set 20–30 câu/2–3 video (lookup, timestamp, multi-hop, "không có trong video"), chạy RAGAS 4 metrics với judge DeepSeek (wrapper n=1) + embeddings bge-m3 local → baseline số liệu.
2. [ ] **Chunking:** parent-child theo thời gian — child ~300 ký tự (≈ 2–4 segment) để retrieve, parent = cửa sổ 1 200 ký tự quanh child để trả lời; giữ `start/end` ở cả hai để vẫn trả link timestamp.
3. [ ] **Search:** giữ Hybrid BM25 + bge-m3 + RRF (đã có), thêm segment tiếng Việt bằng underthesea + bỏ token dấu câu cho BM25.
4. [ ] **Reranking:** thêm `bge-reranker-v2-m3` fp16 (~1.1 GB VRAM, ~100 ms/20 docs), load 1 lần, dùng chung bge-m3 instance cho embedder; nếu đang dubbing chiếm GPU thì fallback CPU.
5. [ ] **Enrichment:** contextual prepend dùng **summary video đã cache** làm ngữ cảnh (rẻ hơn gửi toàn transcript), 1 call/chunk, cache theo hash.
6. [ ] **Query transformation:** decomposition cho câu hỏi nhiều ý (bài học từ failure #1).

### Timeline
- **Tuần 1 (06–12/10/2026):** viết test set + script RAGAS, đo baseline TubeNote hiện tại; thêm latency breakdown từng bước.
- **Tuần 2 (13–19/10/2026):** tích hợp reranker fp16 + chia sẻ model/VRAM với pipeline dubbing; đo lại Context Precision.
- **Tuần 3 (20–26/10/2026):** parent-child chunking theo timestamp + contextual prepend từ video summary; đo Context Recall, Faithfulness.
- **Tuần 4 (27/10–02/11/2026):** query decomposition, ngưỡng rerank score; đưa RAGAS vào CI làm regression gate (không merge nếu metric giảm > 0.05).
