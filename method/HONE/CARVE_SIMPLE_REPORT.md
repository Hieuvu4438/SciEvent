# Báo cáo CARVE-simple và CARVE-simple + HONE

Bộ dữ liệu: **SciEvent** (Dong và cs., EMNLP 2025), split chính thức 1278 / 158 / 163 cửa sổ (train / dev / test).
Bộ chấm điểm: `third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py`, được import nguyên bản, không cài lại.

Mọi số trong báo cáo được tính từ các file dự đoán **đã lưu**. Không có lần huấn luyện hay đánh giá mới nào cho báo cáo này.
Riêng số của CARVE đầy đủ đã được tính lại từ file dự đoán và khớp tuyệt đối với bảng đã công bố. Đây là bước đối chiếu để chắc bộ chấm được dùng đúng.

---

## 0. Tóm tắt

| hệ thống (test) | ROUGE-L F1 | Arg-I IoU F1 | Arg-C IoU F1 |
|---|---|---|---|
| OneIE (baseline tốt nhất về Arg-I/Arg-C của bài gốc) | 72.40 | 53.57 | 41.61 |
| GPT 5-shot (baseline tốt nhất về ROUGE-L của bài gốc) | 75.08 | 49.98 | 34.47 |
| CARVE đầy đủ (3 seed) | 76.93 ± 0.80 | 57.95 ± 2.46 | 50.48 ± 1.15 |
| **CARVE-simple** (3 seed) | **77.22 ± 0.81** | **58.58 ± 0.57** | **50.73 ± 0.19** |
| **CARVE-simple + HONE** (1 seed) | **77.82** | **60.09** | **52.36** |

- **CARVE-simple ≈ CARVE đầy đủ.** Arg-C chênh +0.25, khoảng tin cậy 95% [−1.66, +2.12], không khác biệt. Kiến trúc đơn giản hơn, và phương sai giữa các seed trên Arg-C thấp hơn khoảng 6 lần.
- **CARVE-simple + HONE hơn CARVE-simple cùng seed 42.** Arg-C +1.66 [−0.14, +3.44]: tăng nhưng khoảng tin cậy vẫn chạm 0, nên **chưa có ý nghĩa thống kê**.
- **CARVE-simple + HONE không hơn HONE dùng proposer CARVE đầy đủ (1 seed).** 52.36 so với 53.35, Δ −0.99 [−4.41, +2.17].

---

## 1. Phương pháp

### 1.1 CARVE-simple

CARVE-simple là CARVE bỏ đi hai thành phần mà ablation đơn yếu tố đã chứng minh là **trơ**, tức không tạo khác biệt đo được:

| thành phần | CARVE đầy đủ | CARVE-simple |
|---|---|---|
| Encoder | DeBERTa-v3-large | DeBERTa-v3-large |
| Đầu gán nhãn span | **2 đầu BIO tách rời**: đầu vai trò (9 vai trò chấm điểm) + đầu AAO (Agent, Action, PrimaryObject, SecondaryObject) | **1 đầu BIO gộp** 13 loại span (9 vai trò + 4 loại AAO) |
| Đầu loại sự kiện (4 lớp, mức cửa sổ) | có | có (vẫn cần, vì độ đo nhạy với loại sự kiện) |
| Điều kiện hoá đầu vai trò bằng loại sự kiện dự đoán | **có** (embedding loại cộng vào biểu diễn từ) | **không** |
| Luật giải mã hiệu chỉnh | τ = 0.90, min_len = 3 | τ = 0.90, min_len = 3 (đóng băng lại trên dev, ra đúng giá trị cũ) |

**Huấn luyện** (`method/CARVE-full/configs/carve_simple.json`):
- Tối ưu: 30 epoch, batch 8, AdamW; lr encoder 1e-5, lr head 1e-4; weight decay 0.01, warmup 10%, dropout 0.1, clip gradient 1.0.
- Loss: trọng số vai trò 1.0 (cross-entropy trên nhãn BIO gộp), trọng số loại sự kiện 0.5.
- Không dùng CRF, LLRD hay head vai trò mức span.
- Trọng số lưu ở fp32, tính toán bf16 autocast.
- Checkpoint được chọn trên dev theo Arg-C sau giải mã hiệu chỉnh (quét τ ở mỗi epoch).

**Giải mã:**
- Lấy argmax trên toàn bộ không gian nhãn gộp.
- Span thuộc 9 vai trò chấm điểm trở thành đối số ứng viên. Chỉ giữ span có độ tin cậy (trung bình posterior token) ≥ τ và độ dài ≥ min_len.
- Span Action được dùng làm trigger; Agent / PrimaryObject / SecondaryObject tạo bộ ba cho ROUGE-L. Tất cả lấy từ **cùng một** bộ gán nhãn.
- Loại sự kiện là argmax của đầu loại.

**Quy trình đóng băng** (`method/CARVE-full/scripts/freeze_and_test_simple.py`):
- Chọn một cặp (τ, min_len) sao cho **trung bình** dev Arg-C trên 3 seed (42 / 13 / 101) là lớn nhất. Lưới giống hệt CARVE: τ ∈ {0.5 … 0.95}, min_len ∈ {2, 3, 4}, merge_gap = 0.
- Áp một luật duy nhất đó cho cả 3 seed trên test, chấm **một lần**.

### 1.2 CARVE-simple + HONE (1 seed)

HONE là mô hình **đề xuất rồi thẩm định** (propose-then-verify). Proposer là CARVE-simple. Ở đây **không** dùng chiến lược đa seed: chỉ 1 proposer (seed 42) và 1 verifier (seed 42), không gộp ứng viên giữa các seed, không ensemble.

```
cửa sổ ──► CARVE-simple (seed 42) ──► mọi span ứng viên (argmax, KHÔNG ngưỡng) + 17 đặc trưng bằng chứng
                                              │
                                              ▼
        "event type: <T> . w1 … <a> span </a> … wn"  +  đặc trưng proposer
                                              │
                              verifier: cross-encoder DeBERTa-v3-large
                                              │
                                              ▼
                     10 lớp {reject, 9 vai trò} cho từng ứng viên
                                              │
                                              ▼
        điểm giữ + trộn vai trò (α) + loại sự kiện kết hợp (λ) + chống chồng lấn ──► đối số
```

**Bước 1 — Proposer và ứng viên:**
- CARVE-simple giải mã argmax trên nhãn gộp và giữ **mọi** span thuộc 9 vai trò chấm điểm, không lọc ngưỡng.
- Mỗi ứng viên mang 17 đặc trưng:
  - khối vai trò: trung bình P(B-r) + P(I-r) trên span, cho mỗi vai trò r (9 đặc trưng);
  - P(O);
  - độ tin cậy trung bình;
  - độ tin cậy lớn nhất;
  - số seed đồng thuận (ở đây luôn là 1/3);
  - log độ dài;
  - vị trí đầu và cuối tương đối;
  - số mảnh (luôn là 0).
- Trigger, bộ ba AAO và loại sự kiện lấy từ chính proposer.

**Bước 2 — Cross-fitting (bắt buộc; H4 đã chứng minh):**
- Proposer nhớ luôn dữ liệu train: ứng viên in-sample đúng tới 97.5%, nên verifier học trên đó sẽ chỉ biết "giữ hết". Bỏ bước này làm Arg-C giảm 4.3 điểm.
- Cách làm: chia train thành 5 fold ở mức cửa sổ, phân tầng theo loại sự kiện. Với mỗi fold, huấn luyện lại CARVE-simple trên 4/5 còn lại (chọn epoch trên dev thật), rồi giải mã fold bị giữ lại.
- Kết quả: ứng viên out-of-fold có **5.06 ứng viên/cửa sổ, 38.0% đúng**, khớp với dev (5.01/cửa sổ).
- Ứng viên dev/test đến từ CARVE-simple seed 42 huấn luyện trên toàn bộ train: 792 ứng viên dev, 832 ứng viên test.

**Bước 3 — Verifier:**
- Encoder: DeBERTa-v3-large, thêm hai token đặc biệt `<a>` và `</a>` để đánh dấu span.
- Đầu vào: `event type: <loại dự đoán> .` + cửa sổ có span được đánh dấu, tối đa 320 subword.
- Biểu diễn: [h_CLS ; h_<a> ; h_</a> ; MLP(17 đặc trưng) → 64 chiều] → MLP → 10 lớp.
- Nhãn: vai trò của span vàng có IoU lớn nhất nếu IoU > 0.5, ngược lại là `reject`.
- Huấn luyện: 5 epoch, AdamW (lr encoder 1e-5, lr head 1e-4, weight decay 0.01, warmup 10%), batch 8 × tích luỹ 2 = 16, dropout 0.1, gradient checkpointing, bf16 autocast.
- Chọn epoch trên dev: epoch 5.

**Bước 4 — Giải mã:**
- Điểm giữ = 1 − p(reject).
- Phân phối vai trò = trộn tuyến tính giữa phân phối vai trò của verifier (trọng số α) và khối vai trò chuẩn hoá của proposer (trọng số 1 − α).
  Lý do: verifier lọc span tốt hơn (keep AUC 0.890 so với 0.860), nhưng proposer gán vai trò tốt hơn (81.3% so với 77.1%).
- Loại sự kiện = argmax_t [log p(t) + λ·Σ log P(vai trò_i | t)], với P(vai trò | t) ước lượng từ **nhãn vàng của train** (làm trơn add-1).
- Chọn span tham lam theo điểm giữ ≥ θ, độ dài ≥ min_len, bỏ ứng viên chồng lấn với span đã nhận (luật `any`).

**Luật đóng băng trên dev** (`assets/rule_hs_v42_ps42.json`, lúc 2026-09-23 19:31:30, trước khi có bất kỳ dự đoán test nào):

| θ | min_len | chống chồng lấn | λ (loại kết hợp) | α (trọng số vai trò của verifier) |
|---|---|---|---|---|
| 0.20 | 1 | any | 0.5 | 0.25 |

Lưới tìm: α ∈ {0, .25, .5, .75, 1}; λ ∈ {0, .25, .5, .75, 1, 1.5}; chống chồng lấn ∈ {any, iou, wis}; min_len ∈ {1, 2, 3}; θ ∈ {0.20 … 0.80}.

---

## 2. Kết quả test — CARVE-simple

### 2.1 Trigger ROUGE-L (so với Bảng 3 của bài gốc)

| Method | P | R | **F1** |
|---|---|---|---|
| EEQA | 81.93 | 34.57 | 45.05 |
| DEGREE | 64.56 | 63.49 | 56.85 |
| OneIE | 73.73 | 79.40 | 72.40 |
| GPT (0-shot) | 65.38 | 72.73 | 67.57 |
| GPT (1-shot) | 72.67 | 77.77 | 74.05 |
| GPT (2-shot) | 73.38 | 78.45 | 74.76 |
| GPT (5-shot) — tốt nhất của bài gốc | 73.70 | 78.82 | 75.08 |
| Qwen (2-shot) | 57.27 | 69.71 | 61.18 |
| Llama (0-shot) | 54.88 | 61.07 | 55.83 |
| DS-R1-Llama (1-shot) | 41.81 | 41.94 | 40.72 |
| CARVE đầy đủ | 83.85 ± 1.25 | 76.28 ± 0.55 | 76.93 ± 0.80 |
| **CARVE-simple** | **84.77 ± 0.43** | 76.15 ± 1.21 | **77.22 ± 0.81** |
| *Δ CARVE-simple so với GPT 5-shot* | *+11.07* | *−2.67* | *+2.14* |

### 2.2 Trích xuất đối số, IoU > 0.5 (so với Bảng 4 của bài gốc)

| Method | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| OneIE — tốt nhất của bài gốc | 51.11 | 56.29 | 53.57 | 39.69 | 43.71 | 41.61 |
| GPT (0-shot) | 43.03 | 55.56 | 48.50 | 30.40 | 39.25 | 34.26 |
| GPT (1-shot) | 50.14 | 50.22 | 50.18 | 34.60 | 34.66 | 34.63 |
| GPT (2-shot) | 49.12 | 51.29 | 50.18 | 33.99 | 35.49 | 34.72 |
| GPT (5-shot) | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| Qwen (5-shot) | 46.94 | 31.36 | 37.60 | 21.67 | 14.48 | 17.36 |
| Llama (1-shot) | 44.70 | 34.08 | 38.68 | 18.93 | 14.44 | 16.38 |
| DS-R1-Llama (1-shot) | 42.62 | 17.67 | 24.98 | 19.59 | 8.12 | 11.48 |
| CARVE đầy đủ | 63.46 ± 2.16 | 53.35 ± 2.98 | 57.95 ± 2.46 | 55.30 ± 1.13 | 46.47 ± 1.74 | 50.48 ± 1.15 |
| **CARVE-simple** | **62.64 ± 0.23** | **55.03 ± 1.08** | **58.58 ± 0.57** | **54.24 ± 0.46** | **47.65 ± 0.65** | **50.73 ± 0.19** |
| *Δ CARVE-simple so với OneIE* | *+11.53* | *−1.26* | *+5.01* | *+14.55* | *+3.94* | *+9.12* |

### 2.3 Từng seed (luật đóng băng τ = 0.90, min_len = 3)

| seed | RgL-P | RgL-R | RgL-F1 | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 85.13 | 76.78 | 77.82 | 62.91 | 54.41 | 58.35 | 54.66 | 47.28 | 50.70 |
| 13 | 84.29 | 76.92 | 77.55 | 62.50 | 56.29 | 59.23 | 53.75 | 48.41 | 50.94 |
| 101 | 84.88 | 74.76 | 76.30 | 62.50 | 54.41 | 58.17 | 54.31 | 47.28 | 50.55 |
| **trung bình ± đlc** | 84.77 ± 0.43 | 76.15 ± 1.21 | 77.22 ± 0.81 | 62.64 ± 0.23 | 55.03 ± 1.08 | 58.58 ± 0.57 | 54.24 ± 0.46 | 47.65 ± 0.65 | 50.73 ± 0.19 |

*So sánh với CARVE đầy đủ theo từng seed, cột Arg-C F1: 49.22 / 51.48 / 50.75.*

### 2.4 Bốn chế độ so khớp (trung bình ± đlc trên 3 seed)

| chế độ | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|
| Exact Match | 40.09 ± 1.53 | 35.21 ± 0.87 | 37.49 ± 1.12 | 36.60 ± 1.69 | 32.15 ± 0.96 | 34.23 ± 1.27 |
| Simple overlap | 74.89 ± 1.02 | 65.79 ± 0.54 | 70.04 ± 0.17 | 63.99 ± 0.58 | 56.22 ± 0.89 | 59.85 ± 0.43 |
| SciREX > 0.5 | 70.62 ± 1.28 | 62.04 ± 0.22 | 66.05 ± 0.44 | 60.16 ± 1.12 | 52.85 ± 0.29 | 56.26 ± 0.44 |
| **IoU > 0.5** (chỉ số chính) | 62.64 ± 0.23 | 55.03 ± 1.08 | 58.58 ± 0.57 | 54.24 ± 0.46 | 47.65 ± 0.65 | 50.73 ± 0.19 |

*Cùng bảng cho CARVE đầy đủ, cột ArgC-F1: EM 33.43, overlap 59.11, SciREX 55.03, IoU 50.48.*

### 2.5 Dev (luật đóng băng)

| | seed 42 | seed 13 | seed 101 | trung bình ± đlc |
|---|---|---|---|---|
| CARVE đầy đủ, dev Arg-C | 47.40 | 47.15 | 46.87 | 47.14 ± 0.26 |
| **CARVE-simple, dev Arg-C** | 49.19 | 47.39 | 47.72 | **48.10 ± 0.96** |

### 2.6 Ý nghĩa thống kê

Bootstrap ghép cặp theo cửa sổ test: 5,000 lần lấy mẫu lại, cả hai hệ thống dùng cùng tập cửa sổ trong mỗi lần, mỗi bên lấy trung bình 3 seed.

| CARVE-simple so với CARVE đầy đủ | Δ | 95% CI | P(Δ > 0) | 95% CI của CARVE-simple |
|---|---|---|---|---|
| Arg-C IoU F1 | +0.25 | [−1.66, +2.12] | 0.603 | [45.89, 55.49] |
| Arg-I IoU F1 | +0.64 | [−1.27, +2.46] | 0.746 | [54.31, 62.70] |

- Không khác biệt với CARVE đầy đủ.
- Cận dưới khoảng tin cậy của CARVE-simple **vẫn trên** OneIE: 45.89 > 41.61 cho Arg-C, 54.31 > 53.57 cho Arg-I.
- ROUGE-L chưa được bootstrap.

---

## 3. Kết quả test — CARVE-simple + HONE (1 seed)

Tất cả các dòng dùng **proposer seed 42** (và verifier seed 42 nếu có), không gộp seed.

### 3.1 Trigger ROUGE-L

| hệ thống (1 seed) | P | R | **F1** |
|---|---|---|---|
| GPT (5-shot) — tốt nhất của bài gốc | 73.70 | 78.82 | 75.08 |
| CARVE đầy đủ s42 | 82.42 | 76.38 | 76.02 |
| CARVE-simple s42 | 85.13 | 76.78 | 77.82 |
| HONE, proposer CARVE đầy đủ s42 | 82.42 | 76.38 | 76.02 |
| **CARVE-simple + HONE** | **85.13** | **76.78** | **77.82** |

ROUGE-L của HONE **bằng đúng** ROUGE-L của proposer mà nó dùng. Trigger và bộ ba AAO lấy thẳng từ proposer; verifier chỉ quyết định các span vai trò. Vì vậy ROUGE-L **không** phải đóng góp của HONE.

### 3.2 Trích xuất đối số, IoU > 0.5

| hệ thống (1 seed) | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| OneIE — tốt nhất của bài gốc | 51.11 | 56.29 | 53.57 | 39.69 | 43.71 | 41.61 |
| CARVE đầy đủ s42 | 61.86 | 49.91 | 55.24 | 55.12 | 44.47 | 49.22 |
| CARVE-simple s42 | 62.91 | 54.41 | 58.35 | 54.66 | 47.28 | 50.70 |
| HONE, proposer CARVE đầy đủ s42 | 69.36 | 53.10 | 60.15 | 61.52 | 47.09 | 53.35 |
| **CARVE-simple + HONE** | **70.18** | 52.53 | **60.09** | **61.15** | 45.78 | **52.36** |
| *Δ CARVE-simple + HONE so với CARVE-simple s42* | *+7.27* | *−1.88* | *+1.74* | *+6.49* | *−1.50* | *+1.66* |
| *Δ CARVE-simple + HONE so với OneIE* | *+19.07* | *−3.76* | *+6.52* | *+21.46* | *+2.07* | *+10.75* |

Cơ chế nhìn thấy rõ ở P/R: verifier **tăng mạnh precision** (Arg-C P +6.5), đổi lại mất một ít recall (−1.5). Nó hoạt động như một bộ lọc span tốt hơn ngưỡng độ tin cậy.

### 3.3 Bốn chế độ so khớp

| chế độ | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|
| Exact Match | 46.37 | 34.71 | 39.70 | 42.61 | 31.89 | 36.48 |
| Simple overlap | 81.45 | 60.98 | 69.74 | 69.42 | 51.97 | 59.44 |
| SciREX > 0.5 | 76.19 | 57.04 | 65.24 | 65.41 | 48.97 | 56.01 |
| **IoU > 0.5** | 70.18 | 52.53 | 60.09 | 61.15 | 45.78 | 52.36 |

*Để so sánh, CARVE-simple s42 ở cột ArgC-F1: EM 34.21, overlap 59.96, SciREX 56.74, IoU 50.70.*

### 3.4 Dev (luật đóng băng)

| hệ thống (1 seed) | dev Arg-C IoU F1 |
|---|---|
| CARVE đầy đủ s42 (τ 0.90, min_len 3) | 47.40 |
| CARVE-simple s42 (τ 0.90, min_len 3) | 49.19 |
| HONE, proposer CARVE đầy đủ | 49.16 |
| **CARVE-simple + HONE** | **49.16** |
| *CARVE-simple + HONE, chỉ dùng verifier (α = 1, λ = 0, luật lúc huấn luyện)* | *46.69* |

- Trên dev, **verifier không thêm gì** so với ngưỡng của CARVE-simple (49.16 so với 49.19).
- Hai hệ thống HONE 1 seed ra cùng một giá trị dev (219 khớp / 394 dự đoán / 497 vàng). Tôi đã kiểm tra: đây là trùng hợp thật. Mỗi hệ dùng đúng tập ứng viên của nó (795 và 792 ứng viên), không nhầm file.

### 3.5 Ý nghĩa thống kê

Bootstrap ghép cặp theo cửa sổ test, 5,000 lần lấy mẫu lại.

| CARVE-simple + HONE so với … | chỉ số | Δ | 95% CI | P(Δ > 0) |
|---|---|---|---|---|
| CARVE-simple s42 | Arg-C | +1.66 | [−0.14, +3.44] | 0.962 |
| | Arg-I | +1.74 | [−0.65, +4.03] | 0.920 |
| CARVE đầy đủ s42 | Arg-C | +3.14 | [−0.62, +6.83] | 0.951 |
| | Arg-I | **+4.84** | **[+0.86, +8.83]** | 0.992 |
| HONE, proposer CARVE đầy đủ | Arg-C | −0.99 | [−4.41, +2.17] | 0.287 |
| | Arg-I | −0.06 | [−3.72, +3.30] | 0.483 |

Để đối chiếu: HONE với proposer CARVE đầy đủ so với CARVE đầy đủ s42 cho Arg-C **+4.13 [+1.57, +6.74]** và Arg-I **+4.90 [+2.25, +7.60]**. Cả hai có ý nghĩa.

95% CI tuyệt đối của CARVE-simple + HONE: Arg-C [47.28, 57.18], Arg-I [55.12, 64.72]. Cả hai cận dưới đều trên OneIE.

---

## 4. Kết luận

1. **CARVE-simple là cấu hình nên dùng thay cho CARVE đầy đủ.** Kết quả tương đương trên cả ba chỉ số (Arg-C 50.73 so với 50.48, Arg-I 58.58 so với 57.95, ROUGE-L 77.22 so với 76.93), kiến trúc gọn hơn, và ổn định giữa các seed hơn nhiều (đlc Arg-C 0.19 so với 1.15).
2. **HONE cộng thêm cho CARVE-simple nhưng chưa có ý nghĩa thống kê** (+1.66 Arg-C, khoảng tin cậy chạm 0). Trên dev nó không cộng thêm gì.
3. **Proposer đơn giản không làm HONE tốt hơn.** Sau khi qua verifier, cả hai proposer về cùng một mức (52.36 so với 53.35, khoảng tin cậy phủ 0). Proposer tốt hơn thì phần việc còn lại cho verifier ít đi: phần cộng thêm giảm từ +4.13 (CARVE đầy đủ) xuống +1.66 (CARVE-simple).
4. **Cấu hình mạnh nhất vẫn là HONE**:
   - 1 seed với proposer CARVE đầy đủ: Arg-C 53.35;
   - đa seed: Arg-C 53.06 ± 1.04, Arg-I 63.05 ± 0.27, ROUGE-L 78.05.

## 5. Vệ sinh tập test và giới hạn

- **Không hệ thống nào được chọn dựa trên test.** Mỗi hệ thống đóng băng luật trên dev trước lần nhìn test duy nhất của nó.
- **Các lần nhìn test ở đây là lần bổ sung** so với lần đánh giá chính của HONE (stage 11 trong `RESEARCH_LOG.md`). Chúng được chạy theo yêu cầu, và phải được công bố như vậy nếu đưa vào paper.
- **CARVE-simple + HONE chỉ có 1 seed** (proposer 42, verifier 42), nên chưa đo được phương sai giữa các seed. Các khoảng tin cậy ở trên chỉ phản ánh nhiễu lấy mẫu của 163 cửa sổ test.
- **Thiếu bootstrap cho ROUGE-L** (CARVE-simple so với CARVE; ở HONE thì ROUGE-L bằng của proposer theo cấu trúc).
- **Rò rỉ:** không mô hình nào đọc `sent_id`. Hậu tố của ID dự đoán đúng loại sự kiện 94–99% (xem `DECISIONS.md` D9).

## 6. File

Đường dẫn dưới đây là vị trí trong kho SciEvent (bản lưu trữ). Bản phát hành chính thức nằm ở `github.com/Hieuvu4438/CARVE`, với tên file tương ứng:
- `configs/proposer.json` (cấu hình CARVE-simple);
- `scripts/proposer_baseline.py` (đóng băng + test CARVE-simple);
- `assets/proposer_rule.json` và `assets/decoding_rule.json` (hai luật đóng băng);
- `runs/verifier_s42` (verifier, trước đây tên là `hs_v42_ps42`).

| nội dung | đường dẫn |
|---|---|
| Cấu hình CARVE-simple | `method/CARVE-full/configs/carve_simple.json` |
| Đóng băng + test CARVE-simple | `method/CARVE-full/scripts/freeze_and_test_simple.py` |
| Luật đóng băng CARVE-simple | `method/CARVE-full/assets/decoding_rules.carve_simple.json` |
| Dự đoán test CARVE-simple | `method/CARVE-full/preds/test_carve_simple_s{42,13,101}.jsonl` |
| Proposer CARVE-simple trong HONE | `scripts/propose.py --proposer simple`; `configs/proposer_simple.json` |
| Ứng viên | `data/cands/simple_{dev,test}_s42.jsonl`, `data/cands/simple_oof_k{0..4}_s42.jsonl` |
| Verifier | `runs/hs_v42_ps42/` (`train_verifier.py --cand_prefix simple_`) |
| Luật đóng băng CARVE-simple + HONE | `assets/rule_hs_v42_ps42.json` |
| Dự đoán test CARVE-simple + HONE | `preds/test_hs_v42_ps42.jsonl` |
| Bootstrap | `scripts/bootstrap_ci.py` |
| Nhật ký chi tiết | `RESEARCH_LOG.md` (stage 13), `method/CARVE-full/docs/PAPER_NOTES.md` (§5.10) |
