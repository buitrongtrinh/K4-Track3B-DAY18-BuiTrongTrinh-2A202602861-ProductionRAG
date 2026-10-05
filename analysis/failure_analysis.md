# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Bùi Trọng Trinh (2A202602861)
**Khóa:** K4 - Track 3B

> **Cấu hình đánh giá:** LLM generation + RAGAS judge = DeepSeek `deepseek-chat` (n=1, qua wrapper);
> embeddings cho `answer_relevancy` = `BAAI/bge-m3` local (DeepSeek không có embeddings API).
> Điểm tuyệt đối có thể lệch so với judge OpenAI, nhưng Baseline và Production dùng **cùng** cấu hình nên Δ so sánh được.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7667 | **0.8975** | +0.1308 |
| Answer Relevancy | 0.7610 | **0.8443** | +0.0833 |
| Context Precision | 0.7500 | **0.8750** | +0.1250 |
| Context Recall | 0.8500 | **0.9250** | +0.0750 |

- **Naive:** paragraph chunking (51 chunks) → dense-only bge-m3 top-3 → LLM.
- **Production:** hierarchical (26 parents / 119 children) → M5 contextual enrichment (1 call/chunk) → BM25 + Dense + RRF top-20 → bge-reranker-v2-m3 → **top-3 parent khác nhau** → LLM (prompt version-aware, temperature=0).

### Latency breakdown (`reports/latency_report.json`, 20 queries, RTX 4050 6GB)

| Bước (build, 1 lần) | ms |
|---|---|
| Chunking | 81 |
| Enrichment (119 chunks, 8 luồng song song; lần đầu ~27 000 ms, sau đó đọc cache) | 193 |
| Indexing BM25 + Qdrant (gồm load bge-m3; khi chạy qua `main.py` model đã load sẵn → ~1 150) | 13 098 |
| Load reranker | 4 881 |

| Bước (mỗi query) | avg ms | p95 ms |
|---|---|---|
| Hybrid search | 16.0 | 74.4 |
| Rerank (20 children, fp16) | 105.4 | 125.0 |
| Generation (DeepSeek) | 939.6 | 1 518.7 |
| **Tổng** | **1 061.0** | **1 647.8** |

→ ~89% latency nằm ở LLM generation; retrieval + rerank chỉ ~120 ms.

---

## Bottom-5 Failures

### #1 — Multi-hop: phép năm + lương (avg 0.375)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024). Lương Senior (P3-P4): 20–35 triệu VNĐ/tháng.
- **Got:** "…được **18 ngày phép năm** (15 + 3)… Về mức lương cụ thể trong khoảng nào, context không có thông tin → Không tìm thấy thông tin."
- **Worst metric:** answer_relevancy = 0.00 (context_precision 0.00, context_recall 0.50, faithfulness 1.00)
- **Error Tree:** Output sai (thiếu nửa câu trả lời) → Context đúng? **Không** — top-3 = `nghi_phep_nam_v2024` (0.94), `nghi_phep_nam_v2023` (0.61), `nghi_phep_khong_luong` (0.40); thiếu `bang_luong_2024` → Query OK? **Không** — 1 query chứa 2 ý, ý "nghỉ phép" lấn át ý "lương" → lỗi ở **Retrieval/Ranking cho multi-hop**.
- **Root cause:** `bang_luong_2024` CÓ trong hybrid top-20 nhưng reranker chấm cả query (đa phần về phép năm) nên đẩy 2 tài liệu nghỉ phép khác lên trên. LLM thấy thiếu dữ liệu nên trả lời "Không tìm thấy" một phần → RAGAS gắn cờ *noncommittal* → answer_relevancy = 0.
- **Suggested fix:** Query decomposition (LLM tách thành 2 sub-query "phép năm 9 năm thâm niên" + "lương Senior"), retrieve/rerank riêng từng sub-query rồi gộp; hoặc diversify top-k (MMR / tối đa 1 version của cùng chính sách) để nhường slot cho tài liệu khác.

### #2 — Numeric: phạt tạm ứng quá hạn (avg 0.672)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày → quá hạn 5 ngày; phí 2%/tháng × 15tr = 300.000 VNĐ/tháng, pro-rata 5 ngày ≈ 50.000 VNĐ.
- **Got:** "Phạt 2%/tháng… 15.000.000 × 2% = **300.000 VNĐ**…"
- **Worst metric:** faithfulness = 0.40 (context_recall 0.50)
- **Error Tree:** Output sai (bỏ bước tính số ngày quá hạn) → Context đúng? **Có** — `tam_ung.md` rerank 0.99 ở vị trí 1, chứa "15 ngày" và "2%/tháng" → Query OK? **Có** → lỗi ở **Generation (suy luận số)**.
- **Root cause:** LLM áp nguyên mức phí 1 tháng thay vì tính 20 − 15 = 5 ngày quá hạn rồi pro-rata. Context không nói rõ cách tính theo ngày → câu "300.000" bị judge coi là kết luận không có căn cứ; GT chứa ý pro-rata không có trong context nên context_recall cũng chỉ 0.5.
- **Suggested fix:** Prompt bắt buộc tính toán từng bước (xác định mốc → số ngày quá hạn → công thức → kết quả) và nêu rõ giả định khi chính sách không quy định cách tính theo ngày.

### #3 — Version: MFA (avg 0.806)
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có — v2.0 bắt buộc MFA cho email, VPN, hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** "Có. Tất cả nhân viên **bắt buộc** kích hoạt MFA cho email, VPN và các hệ thống nội bộ…" (không nhắc v1.0)
- **Worst metric:** context_recall = 0.50
- **Error Tree:** Output đúng nhưng thiếu ý so sánh → Context đúng? **Một phần** — `mat_khau_v2` (0.99) đúng, nhưng 2 slot còn lại là nhiễu (`mua_sam`, `lam_viec_tu_xa`, rerank ≈ 0.00), không có `mat_khau_v1` → Query OK? Có (query không nhắc phiên bản) → lỗi ở **Retrieval (thiếu tài liệu version cũ)**.
- **Root cause:** `mat_khau_v1` không nói gì về MFA nên không khớp query; GT lại yêu cầu ý "v1.0 không yêu cầu MFA" — loại thông tin "vắng mặt" mà similarity search không bắt được.
- **Suggested fix:** Version linking trong metadata (`supersedes`/`superseded_by`): khi retrieve 1 version thì kéo kèm version liên quan; đồng thời bỏ context có rerank score < ngưỡng (≈0.05) thay vì luôn lấp đủ 3 slot bằng nhiễu.

### #4 — Version: thâm niên (avg 0.814)
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024: từ 3 năm, +1 ngày mỗi 3 năm. v2023 cũ: 5 năm.
- **Got:** "Theo chính sách hiện hành (2024)… **3 năm trở lên**… +1 ngày cho mỗi 3 năm (chính sách 2023 cũ yêu cầu 5 năm)." — **đúng hoàn toàn**.
- **Worst metric:** context_precision = 0.50
- **Error Tree:** Output đúng → Context đúng? **Có**, nhưng thứ tự sai: `nghi_phep_nam_v2023` (0.99) xếp **trên** `nghi_phep_nam_v2024` (0.99) → Query OK? Có → lỗi ở **Ranking (không ưu tiên version hiện hành)**.
- **Root cause:** 2 version gần như giống hệt nhau về nội dung nên cross-encoder chấm ngang nhau; judge coi v2023 ở vị trí 1 là kém hữu ích cho GT → precision@1 giảm. Generation vẫn đúng nhờ prompt version-aware.
- **Suggested fix:** Metadata boost/tie-break theo `effective_date` (M5 đã trích được `version`, `effective_date`): khi rerank score chênh < 0.05, ưu tiên tài liệu hiệu lực mới hơn; hoặc filter mặc định `status = hiện hành` trừ khi query hỏi lịch sử.

### #5 — Approval threshold (avg 0.837)
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Trên 50.000.000 VNĐ → Tổng Giám đốc (CEO).
- **Got:** "…cần **Tổng Giám đốc (CEO)** phê duyệt. Ngoài ra, **do là thiết bị CNTT** nên cần xác nhận của phòng CNTT… 3 báo giá…"
- **Worst metric:** faithfulness = 0.50
- **Error Tree:** Output đúng ý chính nhưng thêm suy diễn → Context đúng? **Có** — `mua_sam.md` vị trí 1 → Query OK? Có → lỗi ở **Generation (over-elaboration)**.
- **Root cause:** Câu hỏi chỉ nói "thiết bị", LLM tự giả định "thiết bị CNTT" rồi thêm quy định đi kèm → các claim này bị judge đánh unsupported. Ngoài ra rerank score của `mua_sam` chỉ 0.50 vì child chunk 256 ký tự **cắt đôi bảng phê duyệt** (dòng "Trên 50 triệu → CEO" tách khỏi header bảng) — parent retrieval đã cứu được câu này (LLM nhận cả bảng).
- **Suggested fix:** Prompt "chỉ trả lời đúng điều được hỏi, không thêm giả định"; chunking table-aware (không cắt giữa bảng markdown, lặp lại header bảng ở mỗi child).

---

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. **Output đúng?** → Không: đúng nửa đầu (18 ngày), nửa sau trả "Không tìm thấy thông tin" → answer_relevancy = 0 (noncommittal).
2. **Context đúng?** → Không: 3 context đều về nghỉ phép (v2024, v2023, không lương); `bang_luong_2024` không lọt top-3 dù có trong hybrid top-20 → lỗi không nằm ở recall của search mà ở **bước chọn top-3 sau rerank**.
3. **Query rewrite OK?** → Không có query rewrite: query 2 ý được xử lý như 1 ý; cross-encoder chấm theo ý chiếm ưu thế ("nghỉ phép").
4. **Fix ở bước:** Query transformation (decomposition thành sub-queries) + diversity khi chọn context (không cho 2 version của cùng 1 chính sách chiếm 2/3 slot).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop (#1) — dự kiến kéo context_recall/answer_relevancy của nhóm multi-hop lên rõ nhất.
- Version-aware ranking: tie-break theo `effective_date` + version linking (#3, #4).
- Prompt tính toán từng bước, cấm suy diễn ngoài câu hỏi (#2, #5).
- Ngưỡng rerank score để bỏ context nhiễu thay vì luôn lấy đủ 3.
