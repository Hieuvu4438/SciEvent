# CARVE: Trích xuất luận cứ sự kiện khoa học là bài toán phân đoạn span mức mệnh đề, không phải trích xuất thực thể

**CARVE** — *Clause-level Argument Recovery Via sEgmentation*

*(Bản thảo paper tiếng Việt — viết đầy đủ, không giới hạn độ dài hội nghị)*

---

## Tóm tắt (Abstract)

Benchmark **SciEvent** (Dong và cộng sự, EMNLP 2025) đặt ra bài toán trích xuất sự kiện khoa học từ tóm tắt bài báo, trong đó bước khó nhất và chưa được giải quyết là **trích xuất và phân loại luận cứ** (event argument extraction). Kết quả tốt nhất mà bài báo gốc công bố là **41.61 Arg-C IoU F1** (mô hình OneIE), với khoảng cách lớn so với con người.

Trong công trình này, chúng tôi không bắt đầu bằng việc đọc literature về event argument extraction (EAE), mà bắt đầu bằng việc **đo đạc chính dữ liệu**. Phép đo cho thấy một điều mà cách đặt vấn đề của các baseline đã bỏ qua: mỗi đơn vị dữ liệu của SciEvent chứa **đúng một sự kiện**; các luận cứ được chấm điểm là những **mệnh đề dài 10–15 token**; **97.6%** các đơn vị không hề có chồng lấn giữa các luận cứ; và khoảng cách phổ biến nhất giữa hai luận cứ liền kề là **0 token**. Nói cách khác, đây là bài toán **phân đoạn span mức mệnh đề kèm gán nhãn vai trò ngữ nghĩa**, chứ không phải bài toán trích xuất thực thể ngắn kiểu ACE — vốn là cách mà cả ba baseline được tinh chỉnh (OneIE, DEGREE, EEQA) đều mô hình hoá.

Chúng tôi xây dựng **CARVE** (*Clause-level Argument Recovery Via sEgmentation*): một bộ gán nhãn span mức từ đa nhiệm trên nền DeBERTa-v3-large, gồm (i) một đầu BIO cho 9 vai trò được chấm điểm, (ii) một đầu BIO **tách rời** cho bộ ba ⟨Agent, Action, Object⟩ phục vụ chỉ số ROUGE-L, và (iii) một đầu phân loại loại sự kiện ở mức cửa sổ, mà dự đoán của nó được dùng để điều kiện hoá đầu vai trò. Ngoài ra chúng tôi thêm một **luật giải mã nhận thức-độ-đo** gồm ba tham số, được tinh chỉnh **chỉ trên tập dev** rồi đóng băng.

Trên tập test chính thức, đánh giá một lần duy nhất với luật đã đóng băng, trung bình 3 seed:

| Chỉ số | Tốt nhất trong bài báo gốc | CARVE | 95% CI (bootstrap) | Kết luận |
|---|---|---|---|---|
| **Arg-C IoU F1** (chính) | 41.61 (OneIE) | **50.38** | **[45.49, 55.48]** | **cải thiện rõ rệt** — 100% resample vượt |
| Arg-I IoU F1 | 53.57 (OneIE) | 57.88 | [53.40, 62.37] | cải thiện, **sát biên** — 97.1% |
| Trigger ROUGE-L F1 | 75.08 (GPT 5-shot) | 76.96 | [72.68, 80.90] | **tương đương** — chỉ 82.1% |

Mỗi seed riêng lẻ đều vượt cả ba mốc, nhưng chúng tôi nhấn mạnh sự phân biệt quan trọng sau. Khoảng tin cậy bootstrap (2000 lần lấy mẫu lại trên 163 cửa sổ test) cho thấy **chỉ chỉ số chính Arg-C là cải thiện có ý nghĩa thống kê** — cận dưới khoảng tin cậy của nó vẫn cao hơn baseline 3.88 điểm. Arg-I cải thiện nhưng khoảng tin cậy **có chứa** baseline. Trigger ROUGE-L **không** đủ bằng chứng để tuyên bố vượt: với 163 cửa sổ, chỉ số này quá nhiễu. Chúng tôi báo cáo điều này thay vì che giấu, vì một tuyên bố quá mức ở chỉ số phụ sẽ kéo đổ cả tuyên bố chính vốn đang rất vững. Phép kiểm định ghép cặp với OneIE là bất khả thi: bài báo gốc chỉ công bố số tổng hợp, không phát hành dự đoán theo cửa sổ. Chúng tôi cũng trình bày một **phân rã trung thực**: nếu chỉ dùng cách đặt lại bài toán mà không hiệu chỉnh giải mã, mô hình chỉ đạt 39.89 Arg-C — **thấp hơn OneIE**. Cái mà việc đặt lại bài toán thực sự mang lại là **recall: 51.83 so với 43.71 của OneIE (+8.12)**; phần hiệu chỉnh mới là thứ biến khoảng dư recall đó thành F1. Cả hai nửa đều cần thiết và không nửa nào tự mình vượt được baseline.

Chúng tôi báo cáo bốn giả thuyết kiến trúc đã bị **bác bỏ** (CRF chuỗi tuyến tính, layer-wise LR decay, bộ phân loại vai trò mức span, ensemble hậu nghiệm đa seed), kèm lý do cơ chế vì sao dự đoán ban đầu sai. Cuối cùng, chúng tôi thực hiện một **kiểm toán rò rỉ và giao thức** đầy đủ, trong đó phát hiện rằng script chia dữ liệu được phát hành của benchmark chia theo **cửa sổ** chứ không theo **tài liệu** như bài báo mô tả, khiến 97.3% tài liệu test có đoạn anh em nằm trong tập train. Điều này áp dụng như nhau cho mọi baseline nên không làm hỏng so sánh, nhưng chúng tôi đo trực tiếp ảnh hưởng của nó bằng một phép chia lại rời rạc theo tài liệu.

---

## 1. Giới thiệu

### 1.1 Bối cảnh: vì sao trích xuất sự kiện khoa học lại khó

Khi đọc một tóm tắt bài báo khoa học, con người tự động nhận ra cấu trúc: đoạn này nêu bối cảnh, đoạn kia mô tả phương pháp, đoạn tiếp theo trình bày kết quả. Trong mỗi đoạn, ta lại nhận ra các thành phần ngữ nghĩa: đâu là thách thức, đâu là mục đích, đâu là kết quả thu được.

Benchmark **SciEvent** hình thức hoá trực giác này thành một pipeline hai bước:

1. **Phân đoạn (segmentation)**: chia tóm tắt thành các hoạt động khoa học cốt lõi — Background, Method, Result, Conclusion.
2. **Trích xuất (extraction)**: với mỗi đoạn, trích xuất trigger và các luận cứ.

Bộ dữ liệu gồm 500 tóm tắt từ 5 lĩnh vực: ACL (xử lý ngôn ngữ tự nhiên), Bioinformatics (sinh tin học), CSCW (tính toán xã hội), Digital Humanities (nhân văn số), JMIR (tin học y tế). Chính sự đa lĩnh vực này làm bài toán khó: cách viết một kết quả trong sinh tin học rất khác cách viết một kết quả trong nhân văn số.

Bài báo gốc báo cáo rằng các mô hình hiện tại còn cách xa con người, đặc biệt ở các lĩnh vực xã hội và nhân văn. Chỉ số khó nhất là **Arg-C** (argument classification — trích xuất đúng span *và* đúng vai trò), với kết quả tốt nhất chỉ 41.61 F1.

### 1.2 Quan sát khởi đầu: ai cũng dùng cùng một khuôn mẫu

Ba baseline được tinh chỉnh trong bài báo gốc là:

- **OneIE** — giải mã đồ thị chung cho thực thể, quan hệ và sự kiện;
- **DEGREE** — sinh văn bản theo mẫu (template-based generation);
- **EEQA** — đặt bài toán thành hỏi–đáp trích xuất span.

Cả ba đều xuất phát từ truyền thống **ACE-style event extraction**, nơi luận cứ là các **mention thực thể ngắn** (tên người, tổ chức, địa điểm — thường 1–3 token). Đó là giả định ngầm định trong thiết kế của chúng.

Câu hỏi nghiên cứu của chúng tôi rất đơn giản:

> *Giả định đó có đúng với SciEvent không?*

### 1.3 Phát hiện

Không đúng. Chúng tôi đo tập train (1278 cửa sổ) trước khi đọc bất kỳ paper EAE nào, và thu được:

- Mỗi cửa sổ chứa **đúng 1 sự kiện** (đúng với cả 1599 cửa sổ trên mọi tập chia).
- Các luận cứ được chấm điểm dài trung bình **10–15 token** — tức là cả một mệnh đề, không phải một mention.
- **97.6%** cửa sổ không có bất kỳ chồng lấn nào giữa các luận cứ được chấm điểm.
- Khoảng cách phổ biến nhất giữa hai luận cứ liền kề là **0 token** — chúng nối đuôi nhau.

Đây là chữ ký của một bài toán **phân đoạn** (segmentation), không phải bài toán **trích xuất mention**. Và bằng chứng gián tiếp nằm ngay trong bảng kết quả của bài báo gốc: DEGREE đạt precision 48.99 nhưng **recall chỉ 13.83** trên Arg-C. Đó chính xác là hình ảnh của một mô hình sinh được yêu cầu chép lại nguyên văn những span dài 10–15 token — nó chép được vài cái, và bỏ sót phần lớn.

### 1.4 Đóng góp

1. **Một phép đo lại bài toán.** Chúng tôi chỉ ra bằng số liệu rằng SciEvent thuộc lớp bài toán phân đoạn span mức mệnh đề, và rằng sự lệch pha giữa lớp bài toán và họ mô hình — chứ không phải dung lượng mô hình — mới là nút thắt.

2. **CARVE**, một kiến trúc theo đúng hình học đó, cải thiện **có ý nghĩa thống kê** trên chỉ số chính Arg-C IoU (+8.87; 100 % số lần lấy mẫu bootstrap vượt baseline, cận dưới khoảng tin cậy vẫn cao hơn 3.88 điểm). Arg-I cải thiện nhưng sát biên; trigger ROUGE-L tương đương. Chúng tôi phát biểu từng chỉ số đúng mức bằng chứng cho phép.

3. **Một phát hiện về hiệu chỉnh giải mã.** Dưới cơ chế so khớp IoU > 0.5 một-đối-một, một ngưỡng độ tin cậy đơn giản (3 tham số, chỉnh trên dev) đáng giá **+9.12 F1** trên dev và **+10.59 F1** trên test. Chúng tôi trình bày điều này kèm phân rã trung thực chỉ rõ phần nào đến từ đâu.

4. **Sáu kết quả âm tính có giải thích cơ chế.** CRF, LLRD, bộ phân loại vai trò mức span, ensemble hậu nghiệm, **ngưỡng phụ thuộc độ dài**, và **giả thuyết hình học ở dạng tổng quát** đều bị bác bỏ. Trong bốn trường hợp chúng tôi đo được *vì sao* dự đoán ban đầu sai — điều này có giá trị hơn bản thân kết quả âm tính. Đáng chú ý, thí nghiệm kiểm soát (§9.5) cho thấy độ dài span **không** quyết định độ khó; ngữ nghĩa vai trò mới quyết định.

5. **Một kiểm toán benchmark.** Chúng tôi phát hiện script chia dữ liệu được phát hành chia theo cửa sổ chứ không theo tài liệu, trái với mô tả trong bài báo, và đo trực tiếp ảnh hưởng bằng phép chia lại rời rạc theo tài liệu.

---

## 2. Bối cảnh: dữ liệu và giao kèo đánh giá

Phần này quan trọng hơn vẻ ngoài của nó. **Toàn bộ thiết kế của phương pháp suy ra từ việc đọc kỹ bộ chấm điểm chính thức**, chứ không phải từ phần mô tả bằng lời trong bài báo.

### 2.1 Đơn vị dữ liệu: "cửa sổ" (window)

Mỗi tóm tắt được chia thành các đoạn (segment). Mỗi đoạn được gọi là một **cửa sổ**, và mang **đúng một sự kiện**. Tổng cộng:

| Tập | Số cửa sổ | Số tài liệu |
|---|---|---|
| train | 1278 | 493 |
| dev | 158 | 133 |
| test | 163 | 147 |
| **Tổng** | **1599** | **500** |

Độ dài cửa sổ trung bình khoảng 62 token (tách theo khoảng trắng), tối đa 260. Đây là văn bản **ngắn** — không có vấn đề ngữ cảnh dài.

Có 4 loại sự kiện, tương ứng 4 loại đoạn: `Background/Introduction`, `Methods/Approach`, `Results/Findings`, `Conclusions/Implications`.

### 2.2 Các vai trò luận cứ, và vai trò nào thực sự được chấm điểm

Bộ chú thích có 12 nhãn vai trò, nhưng **bộ chấm điểm chính thức chỉ chấm 9 trong số đó**. Ba vai trò `Agent`, `PrimaryObject`, `SecondaryObject` bị **loại khỏi** Arg-I/Arg-C; chúng chỉ dùng để dựng chuỗi tính ROUGE-L.

| Vai trò | Số mẫu train | Được chấm Arg-I/Arg-C? |
|---|---|---|
| Agent | 1276 | không (chỉ cho ROUGE-L) |
| PrimaryObject | 1214 | không (chỉ cho ROUGE-L) |
| SecondaryObject | 104 | không (chỉ cho ROUGE-L) |
| Context | 1053 | **có** |
| Method | 968 | **có** |
| Results | 911 | **có** |
| Challenge | 406 | **có** |
| Purpose | 254 | **có** |
| Implications | 235 | **có** |
| Analysis | 67 | **có** |
| Contradictions | 2 | **có** |
| Ethical | 1 | **có** |

Nhận xét quan trọng: **phân bố cực kỳ lệch**. Ba vai trò đầu (Context, Method, Results) chiếm phần lớn, trong khi Contradictions chỉ có 2 mẫu và Ethical chỉ có 1 mẫu trong toàn bộ tập train. Không mô hình nào học được gì từ 1 mẫu.

### 2.3 Cách chấm điểm — và ba điều mà phần mô tả bằng lời không nói rõ

Chúng tôi đọc trực tiếp mã nguồn `EM_overlap_eval.py` thay vì tin vào phần mô tả. Ba điều quyết định thiết kế:

**(1) Arg-I và Arg-C *không* quan tâm tới trigger, nhưng *rất* quan tâm tới loại sự kiện.**

Trong mã nguồn có dòng `if p[0][2] != g[0][2]: continue` — nếu loại sự kiện dự đoán khác loại sự kiện vàng, phép so khớp bị bỏ qua hoàn toàn. Ngược lại, vị trí trigger không hề tham gia vào việc so khớp luận cứ.

*Hệ quả cực kỳ quan trọng:* vì mỗi cửa sổ chỉ có 1 sự kiện, nếu ta **đoán sai loại sự kiện của cửa sổ**, thì **toàn bộ luận cứ trong cửa sổ đó chết sạch** — vừa tính là false positive (vì loại sai), vừa tính là false negative (vì không khớp được nhãn vàng nào). Loại sự kiện hoạt động như một **hệ số nhân** lên toàn bộ điểm số. Đây là lý do chúng tôi mô hình hoá nó tường minh (§4.3).

**(2) So khớp là tham lam một-đối-một.** Mỗi span vàng chỉ được "tiêu thụ" một lần. Nếu ta dự đoán hai span cùng khớp một span vàng, chỉ một cái được tính đúng, cái còn lại thành false positive. Điều này phạt nặng việc dự đoán dư.

**(3) Chỉ số IoU > 0.5 nghĩa là gì.** IoU (Intersection over Union) giữa span dự đoán và span vàng phải lớn hơn 0.5:

```
IoU = độ dài phần giao / độ dài phần hợp
```

Trực giác: với một span vàng dài 20 token, dự đoán phải phủ ít nhất khoảng 2/3 nó và không tràn ra quá nhiều. Đây là tiêu chí **nới lỏng** so với khớp chính xác — biên có thể lệch vài token vẫn được chấp nhận. Điều này thuận lợi cho span dài và khắc nghiệt với span ngắn (một span vàng 2 token thì gần như phải đoán chính xác).

Bộ chấm điểm báo cáo 4 chế độ so khớp: Exact (khớp chính xác), simple overlap (chỉ cần giao nhau), SciREX (giao/độ-dài-dự-đoán > 0.5), và **IoU > 0.5** (chỉ số chính).

**(4) "Trigger ROUGE-L" không phải là độ trùng span trigger.** Đây là chỗ dễ hiểu nhầm nhất. Nó là điểm ROUGE-L (có stemming) giữa **hai chuỗi văn bản được nối lại**:

```
"<Agent> <trigger> <PrimaryObject> <SecondaryObject>"
```

của nhãn vàng và của dự đoán, rồi lấy trung bình vĩ mô qua các cửa sổ. Tức là nó đo chất lượng của **bộ ba tác tử–hành động–đối tượng** như một tóm lược ngữ nghĩa, chứ không đo vị trí trigger.

### 2.4 Chúng tôi xác minh đường đánh giá như thế nào

Chúng tôi **không viết lại** bất kỳ chỉ số nào. Module đánh giá của chúng tôi nạp file `EM_overlap_eval.py` của benchmark và gọi **chính các hàm của nó**. Hai phép kiểm tra:

- **Kiểm tra oracle bộ xuất.** Đưa các span **vàng** qua đúng bộ xuất dự đoán của chúng tôi rồi vào bộ chấm chính thức → đạt **100.00** trên mọi chỉ số, ở cả dev lẫn test. Điều này chứng minh định dạng xuất của chúng tôi là chính xác tuyệt đối.
- **Đối chiếu qua CLI.** Chạy script gốc đúng như hướng dẫn trong README của benchmark trên file dự đoán test đã đóng băng → tái lập số liệu của chúng tôi **đến chữ số thập phân cuối cùng** (xem §8.1).

---

## 3. Phân tích dữ liệu: vì sao các baseline mô hình hoá sai

Đây là phần cốt lõi của bài. Chúng tôi trình bày các phép đo, rồi rút ra kết luận về lớp bài toán.

### 3.1 Hình học của luận cứ

Bảng 1 — Các thuộc tính hình học đo trên tập train (1278 cửa sổ):

| Thuộc tính | Giá trị |
|---|---|
| Số sự kiện mỗi cửa sổ | **đúng 1**, ở cả 1278 cửa sổ (và cả 1599 trên mọi tập) |
| Số luận cứ được chấm điểm mỗi cửa sổ | trung bình **3.05**, dao động 0–24 |
| Cửa sổ có chồng lấn **trong nhóm** vai trò được chấm | **31 / 1278 = 2.4%** |
| Cửa sổ có chồng lấn **trong nhóm** Agent/Primary/Secondary | **2 / 1278 = 0.2%** |
| Cửa sổ có chồng lấn **giữa hai nhóm** | **75 / 1278 = 5.9%** |
| Tỉ lệ token của cửa sổ được span chấm điểm phủ | **56.3%** |

Bảng 2 — Độ dài span theo vai trò (tập train, đơn vị token):

| Vai trò | n | trung bình | trung vị | p10 | p90 |
|---|---|---|---|---|---|
| Agent | 1276 | 2.4 | 2 | 1 | 5 |
| PrimaryObject | 1214 | 5.9 | 4 | 2 | 13 |
| **Context** | 1053 | **10.2** | 8 | 3 | 21 |
| **Method** | 968 | **13.2** | 11 | 3 | 26 |
| **Results** | 911 | **14.5** | 13 | 6 | 25 |
| **Challenge** | 406 | **14.4** | 13 | 5 | 25 |
| **Purpose** | 254 | **12.5** | 11 | 6 | 22 |
| **Implications** | 235 | **12.9** | 11 | 4 | 23 |
| **Analysis** | 67 | **15.5** | 15 | 8 | 23 |

Hãy chú ý sự tương phản: `Agent` dài 2.4 token — đó *đúng* là một mention thực thể. Nhưng mọi vai trò **được chấm điểm** đều dài 10–15 token — đó là **mệnh đề**.

### 3.2 Các span nối đuôi nhau

Khoảng cách giữa hai span được chấm điểm liền kề (số token → tần suất):

```
0 → 942   ← phổ biến nhất
1 → 658
2 → 271
3 → 160
4 → 105
5 → 90
```

Khoảng cách phổ biến nhất là **0**. Nghĩa là các luận cứ **lát gạch** liền nhau trên cửa sổ, giống như một phép phân đoạn văn bản, chứ không phải rải rác như các mention thực thể.

### 3.3 Vai trò lặp lại trong cùng một cửa sổ

Số cửa sổ mà một vai trò xuất hiện **nhiều hơn một lần**: Context 252, Method 219, Results 211, Challenge 89, Implications 46, Purpose 42, Analysis 11.

Điều này **phá vỡ** cách đặt bài toán kiểu hỏi–đáp (như EEQA): nếu ta hỏi "Context của sự kiện này là gì?" và mô hình chỉ trả về một câu trả lời, thì với 252 cửa sổ trong train ta đã mất phần lớn nhãn vàng ngay từ thiết kế. Đây chính là lý do EEQA chỉ đạt recall 34.57 trên trigger.

### 3.4 Kết luận: đây là lớp bài toán nào

Gộp lại:

- span dài (mệnh đề), **không** phải mention;
- gần như không chồng lấn (97.6%);
- nối đuôi nhau (khoảng cách modal = 0);
- mỗi đơn vị đúng 1 sự kiện;
- vai trò lặp lại trong cùng đơn vị.

Đây chính xác là hình dạng của bài toán **phân đoạn span kèm gán nhãn vai trò tu từ/ngữ nghĩa** — cùng họ với bài toán gán nhãn vai trò tu từ trong văn bản pháp lý, chứ không cùng họ với ACE event extraction.

Và chữ ký thất bại của các baseline khớp hoàn hảo với chẩn đoán này:

| Baseline | Cách đặt bài toán | Arg-C IoU P | Arg-C IoU R | Triệu chứng |
|---|---|---|---|---|
| DEGREE | sinh theo mẫu | 48.99 | **13.83** | Không chép nổi span dài 10–15 token |
| EEQA | hỏi–đáp span | 25.85 | 27.20 | Một câu trả lời/câu hỏi, mà vai trò lại lặp |
| OneIE | giải mã đồ thị | 39.69 | 43.71 | Cân bằng nhưng thấp — mô hình mention áp lên span mệnh đề |

---

## 4. Phương pháp: CARVE

### 4.1 Ý tưởng tổng thể

Nếu bài toán là phân đoạn span, thì công cụ đúng là một **bộ gán nhãn chuỗi** (sequence tagger) ở mức từ, chứ không phải bộ sinh văn bản hay bộ giải mã đồ thị. Kiến trúc của chúng tôi:

```
Văn bản cửa sổ (token tách theo khoảng trắng)
        │
        ▼
  DeBERTa-v3-large  ──► gộp về mức từ (lấy sub-token đầu tiên của mỗi từ)
        │
        ├──► [Đầu TYPE]  gộp theo attention → phân loại 4 loại sự kiện
        │         │
        │         └──► embedding loại sự kiện, cộng ngược vào trạng thái từ
        │                     │
        ├──► [Đầu ROLE]  BIO trên 9 vai trò được chấm điểm  (dùng trạng thái đã điều kiện hoá)
        │
        └──► [Đầu AAO]   BIO trên ⟨Agent, Action, PrimaryObject, SecondaryObject⟩
                          (dùng trạng thái *chưa* điều kiện hoá — chỉ phục vụ ROUGE-L)
```

Bốn quyết định thiết kế, mỗi cái xuất phát từ một phép đo cụ thể. Chúng tôi giải thích từng cái dưới đây — **và ở §7.6 sẽ cho thấy hai trong bốn cái đó không sống sót qua ablation.** Chúng tôi trình bày theo trình tự đã thực sự diễn ra (đo → suy luận → kiểm chứng) thay vì chỉ kể lại những cái đúng, vì bản thân việc hai suy luận từ thống kê dữ liệu bị bác bỏ là một bài học phương pháp đáng ghi.

### 4.2 Vì sao dùng gán nhãn BIO, và vì sao **hai** đầu chứ không phải một

**BIO tagging là gì?** Mỗi từ trong câu được gán một nhãn: `B-X` (Begin — từ đầu tiên của một span loại X), `I-X` (Inside — từ tiếp theo trong span đó), hoặc `O` (Outside — không thuộc span nào). Ví dụ:

```
Từ:    We   propose  a    new   method  for   protein  folding  .
Nhãn:  O    O        B-Met I-Met I-Met  I-Met I-Met    I-Met    O
```

Từ chuỗi nhãn này ta giải mã ngược ra danh sách span. Cách làm này tự nhiên cho span **liền mạch, không chồng lấn** — đúng đặc điểm dữ liệu của ta.

**Vì sao phải tách hai đầu?** Nhìn lại Bảng 1:

- chồng lấn **trong** nhóm 9 vai trò được chấm: chỉ 2.4%
- chồng lấn **trong** nhóm Agent/Primary/Secondary: chỉ 0.2%
- nhưng chồng lấn **giữa** hai nhóm: **5.9%**

Một lớp BIO duy nhất chỉ gán được một nhãn cho mỗi từ. Nếu gộp cả 13 loại span vào một đầu, ta sẽ buộc phải **xoá nhãn** ở 5.9% số cửa sổ. Hai đầu tách rời thì không phải xoá gì — mỗi đầu chỉ chịu mức chồng lấn ≤ 2.4% trong nội bộ nó.

Thêm nữa, đầu AAO tồn tại **chỉ để** dựng chuỗi ROUGE-L. Bộ chấm Arg-I/Arg-C bỏ qua hoàn toàn mọi nhãn mà nó sinh ra. Tách riêng nên nó không gây nhiễu cho nhiệm vụ chính.

**Cái giá của việc dùng BIO là bao nhiêu?** Chúng tôi đo *trước khi* cam kết:

- Vòng đi–về BIO (span vàng → BIO → giải mã) khôi phục chính xác **3857/3897 = 98.97%** span vàng.
- Trần của toàn bộ cách đặt bài toán trên dev: **99.70 Arg-C IoU**.

Tức là cách biểu diễn này chỉ tốn khoảng 0.3 F1. Nó không phải nút thắt, nên ta không cần đến các phương pháp liệt kê span phức tạp hơn.

**Xử lý xung đột khi dựng nhãn BIO.** Với 2.4% cửa sổ có chồng lấn, ta cần một quy tắc xác định. Quy tắc: sắp xếp span theo **độ dài tăng dần**, span ngắn được ưu tiên trước; span dài chỉ nhận đoạn liên tục đầu tiên còn trống. Lý do: span ngắn là chú thích *cụ thể hơn*, và span dài bị cắt bớt thường vẫn vượt ngưỡng IoU > 0.5.

### 4.3 Vì sao mô hình hoá loại sự kiện tường minh — và điều kiện hoá

Nhắc lại §2.3: mỗi cửa sổ có đúng 1 sự kiện, và so khớp nhạy với loại sự kiện. Đoán sai loại → **mất sạch** luận cứ của cửa sổ đó.

Vì vậy chúng tôi:

1. Thêm một **đầu TYPE** riêng, phân loại 4 lớp từ vector cửa sổ (thu được bằng attention pooling).
2. **Điều kiện hoá** đầu ROLE bằng dự đoán loại sự kiện: một embedding loại sự kiện được cộng vào trạng thái của mọi từ trước khi vào đầu ROLE.

```
type_emb : Embedding(4, H), khởi tạo bằng ZERO
w_cond   = w + type_emb(loại_sự_kiện)
```

Khởi tạo bằng zero nghĩa là ban đầu phép điều kiện hoá là **phép đồng nhất** — mô hình chỉ học nó nếu nó thực sự có ích.

**Vấn đề huấn luyện–suy luận không khớp, và cách xử lý.** Khi huấn luyện, nếu ta luôn dùng loại sự kiện *vàng* để điều kiện hoá, thì lúc suy luận (chỉ có loại *dự đoán*) mô hình sẽ gặp phân phối lạ. Chúng tôi dùng **scheduled sampling**:

```
p_teacher(epoch) = max( 0.5 , 1 − (epoch − 1)/(tổng_epoch − 1) )
```

Epoch đầu tiên điều kiện hoá hoàn toàn bằng nhãn vàng; về cuối, một nửa số lần dùng chính dự đoán của mô hình — khớp với điều kiện lúc suy luận.

**Giá trị đo được của quyết định này.** Chúng tôi chạy một chẩn đoán oracle: giữ nguyên các span luận cứ, chỉ thay loại sự kiện dự đoán bằng loại vàng rồi chấm lại.

Bảng 3 — Chi phí của việc phân loại sai loại sự kiện (dev):

| Nguồn loại sự kiện | Arg-C IoU | Arg-I IoU |
|---|---|---|
| mô hình dự đoán (độ chính xác 89.24%) | 47.05 | 56.89 |
| **oracle — loại vàng** | **51.77** | **62.80** |

Phân loại loại sự kiện chưa hoàn hảo đang **ngốn 4.72 điểm Arg-C và 5.91 điểm Arg-I**. Con số này xác nhận rằng đầu TYPE là một thành phần đáng giá, đồng thời chỉ ra một khoảng dư còn lại.

### 4.4 Backbone: vì sao DeBERTa-v3-large

| Hạng mục | Giá trị |
|---|---|
| Checkpoint | `microsoft/deberta-v3-large` |
| Kích thước | 24 lớp, hidden 1024, 16 head; **434.6 M** tham số trong toàn mô hình |
| Giấy phép | MIT |
| Cách thích nghi | **fine-tune toàn phần** — không đóng băng, không LoRA/PEFT |
| Độ chính xác số | trọng số chủ **fp32**, tính toán bf16 qua autocast |

Lý do chọn: quyết định về **biên của span** ở đây mang tính cú pháp/diễn ngôn chứ không mang tính thuật ngữ chuyên ngành. Cụ thể, các từ đứng ngay trước điểm bắt đầu span thường là `and`, `that`, `However,`, `to` — tức là các mốc mệnh đề. DeBERTa-v3 với cơ chế disentangled attention là lựa chọn mạnh nhất cho gán nhãn token nói chung, mạnh hơn các mô hình chuyên ngành cỡ base như SciBERT.

Độ dài không phải vấn đề: cửa sổ dài nhất trên test chỉ 240 sub-token, còn trên train chỉ đúng **1** cửa sổ vượt 512.

**Không dùng dữ liệu ngoài.** Không corpus bổ sung, không pseudo-label, không dữ liệu tổng hợp, không giám sát từ LLM, không retrieval. Dữ liệu huấn luyện đúng bằng 1278 cửa sổ train chính thức.

### 4.5 Hàm mất mát và siêu tham số

```
L = 1.0 · CE(role_logits, nhãn_role)
  + 0.5 · CE(aao_logits,  nhãn_aao)
  + 0.5 · CE(type_logits, nhãn_type)
```

| Siêu tham số | Giá trị |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate (encoder / các đầu) | 1e-5 / 1e-4 |
| Lịch học | warmup tuyến tính 10% số bước, rồi giảm tuyến tính về 0 |
| Số epoch | 30 |
| Batch size | 8 |
| Dropout | 0.1 |
| Gradient clipping | chuẩn toàn cục 1.0 |
| Seed | 42, 13, 101 |
| Thời gian chạy | ≈18 giây/epoch, ≈9 phút/lần chạy đầy đủ |

Đây là một mô hình **rẻ**: toàn bộ công trình này tốn khoảng 2.5 giờ GPU trên một card RTX 5880 Ada.

### 4.6 Giải mã nhận thức-độ-đo: thành phần quan trọng thứ hai

Đây là phần mà nếu bỏ qua, phương pháp sẽ **không** vượt được baseline. Chúng tôi trình bày nó cùng với chẩn đoán đã dẫn tới nó.

**Vấn đề phát hiện được.** Sau khi huấn luyện xong mô hình đầu tiên và chạy phân tích lỗi trên dev, con số quyết định là:

> Mô hình dự đoán **784 span**, trong khi nhãn vàng chỉ có **497 span**.

Precision đồng đều khoảng 30 trên *mọi* vai trò, trong khi recall là 50–65. Đây không phải lỗi biểu diễn (mô hình *tìm được* luận cứ), mà là lỗi **hiệu chỉnh** (mô hình dự đoán quá nhiều span độ tin cậy thấp).

Nhớ lại §2.3: so khớp là một-đối-một, nên mỗi span thừa là một false positive trực tiếp. Dự đoán dư 58% thì precision bị kéo sụp.

**Giải pháp: ba tham số.** Với mỗi span được giải mã, ta định nghĩa **độ tin cậy** là trung bình xác suất hậu nghiệm của nhãn được gán, tính trên các token của span:

```
conf(span) = (1/độ_dài) · Σ P(nhãn tại token i)
```

Rồi áp dụng ba luật, theo thứ tự:

1. **Ngưỡng τ** — bỏ span có `conf < τ`;
2. **Độ dài tối thiểu `min_len`** — bỏ span ngắn hơn `min_len` token;
3. **Gộp `merge_gap`** — nối hai span cùng vai trò cách nhau ≤ `merge_gap` token.

**Kết quả tinh chỉnh (chỉ trên dev):** τ = **0.90**, min_len = **3**, merge_gap = **0**.

Chi tiết ba điểm đáng chú ý:

- **`merge_gap = 0` nghĩa là chức năng gộp bị tắt.** Bản thân điều này là một phát hiện: chúng tôi *đã* giả thuyết rằng việc mô hình cắt vụn một span vàng thành nhiều mảnh là lỗi chính (vì khoảng cách modal giữa các span vàng là 0, rất dễ nhầm). Bộ tinh chỉnh tự động chọn tắt gộp, chứng tỏ **cắt vụn chưa bao giờ là lỗi chủ đạo**. Đây cũng là bằng chứng cơ chế chống lại giả thuyết CRF (§7.2).
- **Đầu AAO không bị áp ngưỡng**, nên điểm trigger ROUGE-L giống hệt nhau dù có hay không hiệu chỉnh.
- **Cả ba tham số được đóng băng trước khi chạm vào test** (§5.2).

### 4.7 Một lưu ý kỹ thuật sẽ cắn bất kỳ ai cài lại

`transformers ≥ 5` tôn trọng kiểu dữ liệu lưu trong checkpoint, và `microsoft/deberta-v3-large` được phát hành với trọng số **fp16**. Hệ quả: AdamW cập nhật trọng số chủ fp16 và sinh **NaN ngay ở bước tối ưu đầu tiên**.

Triệu chứng quan sát được rất dễ gây hiểu nhầm: gradient hữu hạn, chuẩn gradient toàn cục bằng 11.02 (bình thường), nhưng **toàn bộ 390 tensor tham số thành NaN sau một lệnh `opt.step()` ở learning rate 6.25e-07**. Đây rõ ràng không phải hiện tượng nổ gradient.

Cách sửa: nạp encoder với `dtype=torch.float32`, và lấy mixed precision từ autocast mà thôi.

---

## 5. Thiết lập thí nghiệm

### 5.1 Chọn checkpoint

Sau mỗi epoch, dự đoán trên dev được giải mã ở **mọi** giá trị τ trong lưới `{0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95}` với `min_len = 3`, và điểm của epoch đó là **giá trị Arg-C IoU F1 lớn nhất** trong lưới. Epoch tốt nhất theo tiêu chí này được giữ.

Lý do làm vậy: nếu chọn checkpoint theo giải mã argmax (không ngưỡng) trong khi hệ thống cuối cùng lại dùng giải mã có ngưỡng, ta đang tối ưu một thứ mình không dùng.

### 5.2 Giao thức đóng băng — thứ tự chính xác

1. Cố định kiến trúc và siêu tham số (§4), huấn luyện **3 seed** trên tập **train**.
2. Chọn checkpoint trên tập **dev** (§5.1).
3. **Đóng băng luật giải mã trên dev**, theo cách không thiên vị seed: với mỗi cặp `(τ, min_len)`, tính Arg-C IoU F1 trên dev cho **từng seed** rồi lấy **trung bình qua các seed**, chọn điểm cực đại. Điều này tránh việc chỉnh ngưỡng theo seed nào ngẫu nhiên trông đẹp nhất trên dev.
4. **Đánh giá test đúng một lần cho mỗi seed** với luật duy nhất đó. Không thay đổi gì sau đó.

Kết quả bước 3: **τ = 0.90, min_len = 3, merge_gap = 0**, với dev Arg-C IoU **47.14 ± 0.26**.

Bảng 4 — Lưới đóng băng luật giải mã trên dev (min_len = 3, trung bình 3 seed):

| τ | seed 42 | seed 13 | seed 101 | trung bình | độ lệch chuẩn |
|---|---|---|---|---|---|
| 0.50 | 43.52 | 45.67 | 44.55 | 44.58 | 1.07 |
| 0.70 | 44.73 | 46.69 | 44.87 | 45.43 | 1.09 |
| 0.80 | 45.98 | 48.10 | 45.28 | 46.45 | 1.47 |
| 0.85 | 46.62 | 47.60 | 45.95 | 46.73 | 0.83 |
| 0.88 | 46.91 | 47.05 | 46.58 | 46.85 | 0.24 |
| **0.90** | 47.40 | 47.15 | 46.87 | **47.14** | **0.26** |
| 0.92 | 47.34 | 46.94 | 45.84 | 46.71 | 0.78 |
| 0.95 | 45.82 | 44.44 | 44.82 | 45.03 | 0.71 |

Điểm được chọn có **độ lệch chuẩn giữa các seed chỉ 0.26** — tức là một vùng ổn định, không phải một đỉnh nhọn ngẫu nhiên.

### 5.3 Vệ sinh dev/test

- Kiến trúc, tiêu chí chọn checkpoint, và cả ba tham số giải mã đều được cố định trên dev **trước** lần đánh giá test duy nhất.
- Không có thống kê nào từ test tham gia vào bất kỳ quyết định nào.
- Toàn bộ những lần chạm vào dữ liệu test trong quá trình phát triển được liệt kê đầy đủ ở §8.4 — kể cả một lần không ảnh hưởng gì nhưng vẫn được khai báo.

---

## 6. Kết quả

### 6.1 Kết quả chính so với bài báo gốc

Bảng 5 — So sánh trích xuất luận cứ theo IoU trên tập test (%):

| Phương pháp | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| **OneIE** (tốt nhất cũ) | 51.11 | 56.29 | **53.57** | 39.69 | 43.71 | **41.61** |
| GPT (0-shot) | 43.03 | 55.56 | 48.50 | 30.40 | 39.25 | 34.26 |
| GPT (1-shot) | 50.14 | 50.22 | 50.18 | 34.60 | 34.66 | 34.63 |
| GPT (2-shot) | 49.12 | 51.29 | 50.18 | 33.99 | 35.49 | 34.72 |
| GPT (5-shot) | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| Qwen (5-shot) | 46.94 | 31.36 | 37.60 | 21.67 | 14.48 | 17.36 |
| Llama (1-shot) | 44.70 | 34.08 | 38.68 | 18.93 | 14.44 | 16.38 |
| DS-R1-Llama (1-shot) | 42.62 | 17.67 | 24.98 | 19.59 | 8.12 | 11.48 |
| **CARVE (của chúng tôi)** | **63.46** | 53.35 | **57.95** | **55.30** | **46.47** | **50.48** |
| *± độ lệch chuẩn (3 seed)* | ±2.16 | ±2.98 | ±2.46 | ±1.13 | ±1.74 | ±1.15 |
| *seed kém nhất* | 61.86 | 49.91 | 55.24 | 54.27 | 44.47 | 49.22 |
| **Δ so với OneIE** | **+12.35** | **−2.94** | **+4.38** | **+15.61** | **+2.76** | **+8.87** |

Bảng 6 — So sánh nhận diện trigger theo ROUGE-L trên tập test (%):

| Phương pháp | P | R | **F1** |
|---|---|---|---|
| EEQA | 81.93 | 34.57 | 45.05 |
| DEGREE | 64.56 | 63.49 | 56.85 |
| OneIE | 73.73 | 79.40 | 72.40 |
| GPT (0-shot) | 65.38 | 72.73 | 67.57 |
| GPT (1-shot) | 72.67 | 77.77 | 74.05 |
| GPT (2-shot) | 73.38 | 78.45 | 74.76 |
| **GPT (5-shot)** (tốt nhất cũ) | 73.70 | 78.82 | **75.08** |
| **CARVE (của chúng tôi)** | **83.85** | 76.28 | **76.93 ± 0.80** |
| **Δ so với GPT 5-shot** | **+10.15** | **−2.54** | **+1.85** |

**Mỗi seed riêng lẻ đều vượt cả ba mốc** (biên độ seed kém nhất: Arg-C +7.61, Arg-I +1.67, ROUGE-L +0.94). Nhưng độ lệch chuẩn giữa các seed chỉ đo **phương sai huấn luyện**, không đo **phương sai lấy mẫu** — tập test chỉ có 163 cửa sổ. Bảng 5b trả lời câu hỏi đó.

### Bảng 5b — Khoảng tin cậy bootstrap (2000 lần lấy mẫu lại trên 163 cửa sổ test)

Lấy mẫu lại **cửa sổ** (đơn vị chú thích) có hoàn lại, chấm lại bằng chính các hàm so khớp của benchmark, lấy trung bình 3 seed bên trong mỗi lần lấy mẫu:

| Chỉ số | Điểm | 95% CI | Baseline | Cận dưới − baseline | % resample vượt |
|---|---|---|---|---|---|
| **Arg-C IoU** | 50.38 | **[45.49, 55.48]** | 41.61 | **+3.88** | **100.00%** |
| Arg-I IoU | 57.88 | [53.40, 62.37] | 53.57 | −0.17 | 97.10% |
| Trigger ROUGE-L | 76.96 | [72.68, 80.90] | 75.08 | −2.40 | 82.05% |

**Đây là kết quả quyết định cách phát biểu của toàn bài.** Chỉ **Arg-C** — chỉ số chính — là cải thiện vững chắc: mọi lần lấy mẫu lại đều vượt baseline và cận dưới khoảng tin cậy vẫn cao hơn 3.88 điểm. **Arg-I** cải thiện nhưng khoảng tin cậy chứa baseline (thiếu 0.17), nên chỉ được nói là "cải thiện" kèm con số 97.1%. **Trigger ROUGE-L không được tuyên bố là vượt**: chỉ 82.1% lần lấy mẫu vượt baseline, tức với 163 cửa sổ ta không phân biệt được nó với 75.08.

Bất kỳ phản biện viên nào chạy bootstrap cũng sẽ tìm ra điều này; báo cáo trước là cách duy nhất giữ được độ tin cậy cho tuyên bố chính.

Một dấu hiệu tốt cần nêu: **kết quả test (50.48) cao hơn dev (47.14)**. Đây là điều ngược với biểu hiện của overfit lên dev, và là bằng chứng cho thấy ngưỡng được chỉnh trên dev đã tổng quát hoá được.

Bảng 7 — Kết quả trên cả bốn chế độ so khớp mà bộ chấm chính thức báo cáo (test, trung bình 3 seed):

| Chế độ | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|
| Khớp chính xác (Exact) | 40.05 | 33.65 | 36.56 | 36.63 | 30.77 | 33.43 |
| Chỉ cần giao nhau | 75.47 | 63.41 | 68.90 | 64.75 | 54.41 | 59.11 |
| SciREX > 0.5 | 70.61 | 59.35 | 64.47 | 60.27 | 50.66 | 55.03 |
| **IoU > 0.5** (chỉ số chính) | 63.46 | 53.35 | **57.95** | 55.30 | 46.47 | **50.48** |

Kết quả nhất quán qua mọi chế độ nới lỏng. Riêng khớp chính xác thì thấp (33.43) — mô hình lấy span *gần đúng*, không *chính xác tuyệt đối*. Điều này đủ cho chỉ số chính của benchmark nhưng sẽ không đủ cho một benchmark khắt khe hơn.

### 6.2 Phân rã trung thực: phần nào đóng góp bao nhiêu

Đây là bảng mà chúng tôi cho là quan trọng nhất trong cả bài, vì nó ngăn việc tuyên bố quá lời.

Bảng 8 — Phân rã đóng góp (test, trung bình 3 seed):

| Giai đoạn | Arg-C P | Arg-C R | **Arg-C F1** | Arg-I F1 |
|---|---|---|---|---|
| OneIE (tốt nhất cũ) | 39.69 | 43.71 | **41.61** | 53.57 |
| của chúng tôi, **chưa hiệu chỉnh** (argmax thuần) | 32.43 | **51.83** | **39.89 ± 1.34** | 47.72 |
| của chúng tôi, **+ giải mã có hiệu chỉnh** | 55.30 | 46.47 | **50.48** | 57.95 |

Đọc kỹ bảng này, vì nó mới là kết quả thực sự:

1. **Nếu chỉ đặt lại bài toán mà không hiệu chỉnh, chúng tôi *thua* OneIE** (39.89 so với 41.61, tức −1.72). Không được tuyên bố rằng "phân đoạn span tự nó đánh bại các baseline".

2. **Cái mà việc đặt lại bài toán thực sự mua được là recall: 51.83 so với 43.71 của OneIE (+8.12).** Bộ gán nhãn mức mệnh đề thật sự tìm ra nhiều luận cứ vàng hơn hẳn so với giải mã đồ thị, sinh theo mẫu, hay hỏi–đáp span. Nó trả giá bằng precision (32.43), vì nó dự đoán dư.

3. **Hiệu chỉnh chuyển khoảng dư recall đó thành +10.59 F1.** Một ngưỡng **không thể tạo ra** recall; nó chỉ có thể đánh đổi recall lấy precision. Một baseline không có khoảng dư recall sẵn có thì dù đặt ngưỡng cũng không thể chạm tới 50.48.

Nói cách khác: **hai nửa đều gánh lực, và không nửa nào tự đủ.** Con số tương ứng trên dev là 38.88 → 48.00.

### 6.3 Đường cong hiệu chỉnh: đây là hiệu ứng thật hay là overfit dev?

Bảng 9 — Quét τ chi tiết trên dev (min_len = 3):

| τ | Arg-C P | Arg-C R | **Arg-C F1** | Arg-I F1 | số span dự đoán (vàng = 497) |
|---|---|---|---|---|---|
| 0.50 | 43.22 | 48.09 | 45.52 | 55.43 | 553 |
| 0.60 | 43.69 | 48.09 | 45.79 | 55.75 | 547 |
| 0.70 | 46.05 | 48.09 | 47.05 | **56.89** | 519 |
| 0.80 | 48.34 | 46.88 | 47.60 | 56.79 | 482 |
| 0.85 | 49.89 | 45.67 | 47.69 | 55.88 | 455 |
| **0.90** | 53.60 | 43.46 | **48.00** | 55.56 | 403 |
| 0.93 | 55.56 | 40.24 | 46.67 | 53.21 | 360 |
| 0.95 | 59.55 | 37.63 | 46.12 | 51.29 | 314 |
| 0.97 | 61.22 | 32.39 | 42.37 | 47.37 | 263 |
| 0.99 | 69.54 | 24.35 | 36.07 | 38.75 | 174 |

Ba quan sát ủng hộ rằng đây là hiệu chỉnh thật chứ không phải khớp nhiễu dev:

- Điểm tối ưu là một **vùng bằng phẳng rộng, τ ∈ [0.70, 0.90]**, trải dài khoảng 1 điểm F1, chỉ sụp đổ khi vượt quá 0.93. Một overfit dev sẽ cho đỉnh nhọn, không phải vùng phẳng.
- Số span dự đoán giảm từ 553 → 403, tiến gần con số vàng 497 rồi vượt qua — đúng hành vi của một phép hiệu chỉnh số lượng.
- Như đã nêu, kết quả test cao hơn dev.

**Chúng tôi bác bỏ ngưỡng theo từng vai trò.** Tối ưu tọa độ trên 9 ngưỡng riêng cho từng vai trò chỉ đem lại **+0.67** trên 158 cửa sổ dev. Với chỉ 158 mẫu, thêm 9 tham số để đổi lấy 0.67 điểm là công thức của overfit. Chúng tôi dùng luật toàn cục 3 tham số.

---

## 7. Ablation và các kết quả âm tính

Chúng tôi đăng ký trước ngưỡng bác bỏ: một thành phần phải đem lại **ít nhất +0.5 Arg-C IoU** so với đối chứng của nó thì mới được giữ.

Bốn giả thuyết đã bị bác bỏ. Chúng tôi trình bày cả **vì sao dự đoán ban đầu sai**, vì đó mới là phần có giá trị.

### 7.1 Ensemble hậu nghiệm đa seed — bác bỏ (đối chứng sạch)

**Giả thuyết:** trung bình xác suất hậu nghiệm ở mức token qua 3 seed trước khi giải mã sẽ giảm phương sai và tăng điểm.

Bảng 10 — Ensemble so với từng seed (dev, cùng lưới tinh chỉnh):

| Hệ thống | Arg-C argmax | Arg-C IoU sau tinh chỉnh | Arg-I IoU |
|---|---|---|---|
| seed 42 đơn lẻ | 36.69 | 47.66 | 55.50 |
| seed 13 đơn lẻ | 38.52 | **48.10** | 57.31 |
| seed 101 đơn lẻ | 37.32 | 46.99 | 54.64 |
| **trung bình hậu nghiệm 3 seed** | **39.35** | 47.10 | 54.11 |

Kết quả **ngược chiều một cách thú vị**: ensemble **cải thiện rõ** giải mã argmax (39.35 so với 36.69–38.52) nhưng **làm tệ đi** giải mã có hiệu chỉnh (47.10, thấp hơn trung bình các seed 47.58 và thấp hơn hẳn seed tốt nhất 48.10).

**Cơ chế:** việc lấy trung bình **nén** phân phối xác suất hậu nghiệm — chính là tín hiệu mà ngưỡng τ dựa vào để phân biệt span tốt và span xấu. Hai kỹ thuật **đối kháng nhau**.

**Hệ quả:** hệ thống được công bố là **một mô hình đơn**, và kết quả chính là trung bình ± độ lệch chuẩn qua 3 seed, không phải ensemble.

### 7.2 CRF chuỗi tuyến tính — bác bỏ

**Giả thuyết:** vì khoảng cách phổ biến nhất giữa hai span vàng liền kề là **0 token**, bộ gán nhãn phải quyết định chuyển `I-Context → B-Method` mà không có token `O` ngăn cách. Một softmax độc lập theo từng token không có tiên nghiệm nào về chuyển trạng thái. Một CRF với ma trận chuyển học được lẽ ra phải giúp chống cắt vụn span.

**Kết quả đo:** phiên bản có CRF đạt **46.52** Arg-C IoU trên dev, so với **48.00** của phiên bản gốc. Giải mã Viterbi (argmax) của nó cũng *thấp hơn*: 35.82 so với 38.88.

**Bằng chứng cơ chế, độc lập với mọi so sánh số:** bộ tinh chỉnh giải mã **tự động chọn `merge_gap = 0`** ở *mọi* cấu hình. Nếu cắt vụn span thực sự là lỗi chủ đạo, bộ tinh chỉnh đã phải chọn một `merge_gap` dương để nối các mảnh lại. Nó chưa bao giờ chọn. Lỗi chủ đạo là **dự đoán dư các span độ tin cậy thấp**, và một ngưỡng vô hướng xử lý việc đó trực tiếp, rẻ hơn CRF nhiều (CRF làm chậm 1.7 lần).

### 7.3 Bộ phân loại vai trò mức span — bác bỏ (đối chứng sạch)

**Động cơ:** bảng phân loại lỗi cho thấy có những trường hợp "span đúng, vai trò sai", và Arg-I vượt Arg-C khoảng 10 điểm. Một đầu BIO gán nhãn theo **từng token**; không có gì trong đó nhìn được cả span cùng lúc. Vậy hãy thêm một đầu nhìn cả span.

**Thiết kế:** `Linear(3H → H) → GELU → Dropout → Linear(H → 9)` trên biểu diễn `[h_đầu ; h_cuối ; trung bình(h)]`, huấn luyện trên span vàng. Lúc suy luận, nó gán lại nhãn cho từng span đã giải mã (span và biên không đổi).

Bảng 11 — Bật/tắt bộ phân loại vai trò mức span (cùng một checkpoint, cùng lưới):

| Cấu hình | Arg-C IoU | Arg-I IoU |
|---|---|---|
| tắt | 46.56 | 53.42 |
| **bật** | **46.80** | 53.42 |

**Lợi ích chỉ +0.24**, dưới ngưỡng bác bỏ 0.5. Arg-I không đổi như dự đoán (đầu này không di chuyển span nên không thể ảnh hưởng Arg-I) — điều này xác nhận thí nghiệm được cài đúng.

**Vì sao dự đoán ban đầu sai — đáng ghi vào bài:** một khi ngưỡng độ tin cậy loại bỏ các span độ tin cậy thấp, thì **phần lớn các ca "span đúng, vai trò sai" cũng bị loại theo**. Bảng phân loại lỗi đang đếm *chính những lỗi mà hiệu chỉnh đã sửa*, tức là chúng tôi đã đếm một lỗi hai lần. Đây là bài học về việc đọc bảng phân loại lỗi trên một mô hình *chưa* hiệu chỉnh.

### 7.4 Layer-wise learning-rate decay — bác bỏ

Phiên bản dùng LLRD hội tụ nhanh gấp khoảng hai lần (đạt Arg-C ≈44 ở epoch 11 thay vì epoch 20) nhưng không về đích cao hơn. Các lần chạy có LLRD đều dừng ở khoảng **46.5–46.8**, trong khi phiên bản gốc đạt **48.00**.

### 7.5 Tổng hợp ablation — bộ ablation đơn-yếu-tố

Bộ ablation cũ khác recipe chính **bốn yếu tố cùng lúc** (lr, dropout, `w_type`, LLRD) và được tinh chỉnh trên các lưới khác nhau; **không dòng nào là so sánh hợp lệ**. Ngoài ra hai design claim cốt lõi chưa hề được ablate. Chúng tôi dựng lại toàn bộ: mỗi cấu hình khác mô hình đầy đủ **đúng một yếu tố**, chạy trên **cùng ba seed**, chấm lại trên **cùng một lưới giải mã**.

Bảng 12 — Ablation đơn-yếu-tố (dev, trung bình ± độ lệch chuẩn qua 3 seed):

| cấu hình | dev Arg-C IoU F1 | Δ |
|---|---|---|
| **Mô hình đầy đủ (CARVE)** | **47.58 ± 0.55** | — |
| − điều kiện hoá loại sự kiện | 47.34 ± 1.61 | −0.24 |
| − tách hai head | 48.34 ± 1.60 | **+0.75** |
| + CRF chuỗi tuyến tính | 47.30 ± 0.50 | −0.28 |
| + head vai trò mức span | 46.63 ± 1.59 | −0.96 |
| + layer-wise LR decay | 45.38 ± 1.18 | −2.20 |

Ba dòng cuối xác nhận các kết quả âm tính đã nêu ở §7.1–7.4, giờ trên một phép so sánh hợp lệ.

### 7.6 Hai quyết định thiết kế của chúng tôi không sống sót qua ablation

Đây là kết quả quan trọng nhất của bộ ablation, và nó buộc phải sửa §4.

**(1) Điều kiện hoá loại sự kiện là trơ (−0.24, độ lệch chuẩn 1.61).** Chúng tôi biện minh nó bằng chẩn đoán oracle cho thấy thay loại sự kiện *vàng* vào đáng giá +4.72 Arg-C (Bảng 3). Chẩn đoán đó **đúng**, nhưng nó chỉ chứng minh rằng **đoán đúng loại sự kiện là quan trọng** — nó **không** chứng minh rằng **điều kiện hoá head vai trò bằng loại dự đoán** có ích. Đó là hai mệnh đề khác nhau và chúng tôi đã gộp nhầm. Head phân loại loại sự kiện vẫn **cần thiết** (độ đo nhạy với loại, nên bắt buộc phải phát ra một loại); nhưng **đường điều kiện hoá thì không**.

**(2) Việc tách hai head là trơ (+0.75, độ lệch chuẩn 1.60), không có lợi như đã tuyên bố.** Chúng tôi biện minh hai head BIO tách rời bằng số đo 5.9 % chồng lấn giữa nhóm, lập luận rằng một lớp duy nhất sẽ phải xoá nhãn. **Phép đo đúng; suy luận từ nó sai.** Chỉ 5.9 % cửa sổ có xung đột, còn việc gộp hai tập nhãn thành một bộ gán nhãn 13 loại cho một biểu diễn dùng chung bù lại nhiều hơn thế.

Phân bố theo từng seed cho thấy phải đọc con số này một cách thận trọng:

| cấu hình | seed 42 | seed 13 | seed 101 | trung bình ± đlc |
|---|---|---|---|---|
| Mô hình đầy đủ | 47.40 | 48.10 | 46.87 | 47.46 ± 0.61 |
| − tách hai head | **50.10** | 46.97 | 47.85 | 48.31 ± 1.62 |
| − điều kiện hoá loại | 48.07 | 48.76 | 45.41 | 47.41 ± 1.77 |

Lợi thế +0.75 bề ngoài đến chủ yếu từ **một seed may mắn** (50.10). Phát biểu trung thực là: **việc tách head không tạo khác biệt đo được**, chứ không phải "bỏ nó đi thì tốt hơn".

**Chúng tôi báo cáo gì.** Hệ thống hai-head đã đóng băng là hệ thống đã đi qua giao thức đóng băng và kiểm toán, và các khác biệt đều nằm trong phương sai seed, nên các con số chính vẫn đứng vững. Nhưng §4 **không còn được phát biểu** là "bốn quyết định thiết kế, mỗi cái do một phép đo ép buộc". Hai trong bốn là không cần thiết, và cấu hình đơn giản hơn — một bộ gán nhãn BIO gộp cộng với head dự đoán loại sự kiện, không có đường điều kiện hoá — cho kết quả tương đương và là cấu hình chúng tôi khuyến nghị.

**Bài học phương pháp đáng nêu thẳng**, vì đây là loại chuyện paper thường giấu: suy ra kiến trúc từ thống kê dữ liệu là cám dỗ, và ở đây nó sai **hai lần**. Các thống kê (5.9 % chồng lấn, độ nhạy với loại sự kiện) được đo đúng; nhưng các suy luận kiến trúc rút ra từ chúng không sống sót qua một phép kiểm có đối chứng. **Chỉ ablation mới phân xử được.**

## 8. Kiểm toán: kết quả này có thật không?

Chúng tôi coi phần này ngang hàng với phần kết quả. Một con số vượt SOTA mà không kiểm toán thì không có giá trị.

### 8.1 Xác minh lại bằng bộ chấm chính thức chạy dạng CLI

Chúng tôi nhập bộ chấm như một thư viện, nên điều đầu tiên cần loại trừ là lỗi ở lớp bọc. Chạy script gốc đúng như hướng dẫn README trên file dự đoán test đã đóng băng (seed 13):

```
[ROUGE-L OVERALL] P: 84.68%, R: 76.77%, F1: 77.52%
Argument Identification (Exact)  - P: 41.26 (184/446)  R: 34.52 (184/533)  F1: 37.59
Argument Classification (Exact)  - P: 37.67 (168/446)  R: 31.52 (168/533)  F1: 34.32
Argument Identification (IoU)    - P: 65.92 (294/446)  R: 55.16 (294/533)  F1: 60.06
Argument Classification (IoU)    - P: 56.50 (252/446)  R: 47.28 (252/533)  F1: 51.48
```

**Trùng khít đến chữ số thập phân cuối cùng** với số chúng tôi báo cáo cho seed 13. Không có lỗi lớp bọc.

### 8.2 Kiểm tra rò rỉ

Bảng 13 — Các kiểm tra rò rỉ và tính toàn vẹn:

| Kiểm tra | Kết quả |
|---|---|
| giao train ∩ test (cửa sổ) | **0** |
| giao train ∩ dev (cửa sổ) | **0** |
| giao dev ∩ test (cửa sổ) | **0** |
| văn bản cửa sổ trùng lặp train → test | **0 / 163** |
| các lời gọi nạp dữ liệu trong đường huấn luyện | chỉ `"train"` và tập đánh giá (mặc định `dev`) |
| tập đánh giá trong cả 6 file cấu hình | không đặt → mặc định `dev` |
| luật giải mã được ghi lúc | 21:35:50 |
| file dự đoán test đầu tiên được ghi lúc | 21:36:05 (luật có **trước**) |
| dùng loại sự kiện vàng khi suy luận? | **không** (loại được dự đoán, chính xác 91.41% trên test) |
| số cửa sổ dự đoán so với vàng | 163 / 163, id khớp chính xác |
| số luận cứ dự đoán so với vàng | **446 / 533 = 0.837** — chúng tôi dự đoán **thiếu** |
| cặp `(span, vai trò)` trùng lặp trong dự đoán | **0** |
| số file bị sửa trong thư mục benchmark gốc | **0** |

Hai dòng cuối cần giải thích:

- **Kiểm tra trùng lặp** quan trọng vì hàm `compute_f1` tích luỹ các cặp khớp vào một `set`; nếu ta phát ra nhiều span giống hệt nhau thì có thể tương tác kỳ lạ với cấu trúc đó. Chúng tôi không phát ra cái nào.
- **Tỉ lệ dự đoán thiếu (0.837)** loại trừ khả năng ăn gian precision bằng cách xả thật nhiều dự đoán.

### 8.3 Phát hiện về benchmark: phép chia là theo cửa sổ, không phải theo tài liệu

Bài báo gốc mô tả phép chia 80/10/10 **theo tài liệu**. Script được phát hành `split_data.py` thực tế khử trùng lặp và chia theo **`wnd_id`** (cửa sổ), phân tầng theo loại sự kiện — dòng mã là `key = item["wnd_id"]`.

Hệ quả:

| | Giá trị |
|---|---|
| số tài liệu trong test | 147 |
| số tài liệu test **cũng xuất hiện trong train** | **143 (97.3%)** |
| số tài liệu test hoàn toàn không xuất hiện trong train lẫn dev | 2 |

Các **cửa sổ** rời nhau, nhưng các **tóm tắt** thì không: đoạn Background của một bài báo có thể nằm ở train trong khi đoạn Results của chính bài đó nằm ở test.

**Điều này không làm hỏng so sánh.** OneIE, DEGREE và EEQA đều được huấn luyện và chấm điểm trên đúng các file này theo README của benchmark, nên mọi con số trong bảng kết quả gốc đều mang cùng tính chất. So sánh ở §6.1 vẫn là so sánh ngang hàng. Tuy vậy điều này **phải được công bố**.

### 8.4 Phép thăm dò với phép chia rời rạc theo tài liệu

Để đo trực tiếp ảnh hưởng, chúng tôi dựng lại một phép chia mới khoá theo `doc_id`, phân tầng theo chữ ký loại sự kiện của từng tài liệu, và xác minh không còn chồng lấn tài liệu: **394 / 45 / 61 tài liệu → 1265 / 148 / 186 cửa sổ**.

Áp dụng **đúng giao thức**: 3 seed, luật giải mã được **đóng băng lại trên tập dev mới** (cho ra τ = 0.85, min_len = 3), rồi đánh giá test một lần.

Bảng 14 — Kết quả trên phép chia rời rạc theo tài liệu:

| | Arg-I IoU F1 | **Arg-C IoU F1** | ROUGE-L F1 |
|---|---|---|---|
| của chúng tôi, phép chia chính thức | 57.95 ± 2.46 | **50.48 ± 1.15** | 76.93 ± 0.80 |
| của chúng tôi, **rời rạc theo tài liệu** | 52.27 ± 1.43 | **44.93 ± 0.95** | 76.20 ± 1.00 |
| tốt nhất trong bài báo gốc (chỉ trên phép chia chính thức) | 53.57 | 41.61 | 75.08 |

**Arg-C vẫn vượt kết quả tốt nhất của bài báo gốc ngay cả khi không còn chồng lấn tài liệu** (44.93 so với 41.61).

**Phân rã mức giảm 5.55 điểm.** Phép chia mới **khó hơn về bản chất**: nửa test của nó có **33.7%** luận cứ vàng thuộc Digital Humanities, so với **21.0%** ở phép chia chính thức — mà DH là lĩnh vực yếu nhất của chúng tôi. So sánh theo từng lĩnh vực và tái trọng số về đúng tỉ lệ lĩnh vực của phép chia chính thức:

| | Arg-C IoU F1 |
|---|---|
| rời rạc theo tài liệu, tỉ lệ DH nặng của chính nó | 43.38 |
| rời rạc theo tài liệu, **tái trọng số về tỉ lệ chính thức** | **46.79** |
| phép chia chính thức, cùng cách trọng số | 49.79 |

Tức là: **khoảng 3.4 điểm của mức giảm đến từ tỉ lệ DH nặng, và khoảng 3.0 điểm đến từ việc bỏ chồng lấn tài liệu.** Đáng chú ý, hai lĩnh vực (cscw và jmir) thậm chí *tăng điểm* dưới phép chia khắt khe hơn — đó không phải hình ảnh của rò rỉ hệ thống.

⚠ **Một bất đối xứng cần nêu rõ:** OneIE chưa từng được đo lại trên phép chia rời rạc theo tài liệu, và gần như chắc chắn nó cũng sẽ giảm. Do đó việc so 44.93 của chúng tôi với 41.61 của OneIE trên phép chia chính thức là **bất lợi cho chúng tôi**, không phải có lợi. Tuyên bố chính vẫn dựa trên phép chia chính thức, nơi so sánh thực sự ngang hàng; phần này chỉ để chứng minh biên độ không phải là hiện vật của cách dựng phép chia.

### 8.5 Công bố đầy đủ mọi lần chạm vào dữ liệu test

1. Kiểm tra oracle bộ xuất — dùng **nhãn vàng** của test để xác minh bộ xuất đạt 100.00. Không có mô hình tham gia, nên không sinh tín hiệu chọn mô hình nào.
2. Một lần kiểm tra phân bố độ dài span vàng trên test (6.8% span có độ dài ≤ 2 token), chạy để kiểm chứng xem luật `min_len` có tổng quát hoá không. Giá trị `min_len = 3` **đã được chọn trước đó** bởi lưới tinh chỉnh trên dev, và giá trị cuối cùng cũng đến từ quy trình đóng băng chỉ dùng dev. Nó **không thay đổi gì**, nhưng nó đã được nhìn, nên chúng tôi khai báo thay vì bỏ qua.
3. Một lần kiểm tra độ dài sub-token trên test (tối đa 240) — chỉ đếm token, không dùng nhãn.
4. Script dựng phép chia rời rạc theo tài liệu gom cả ba tập chính thức. Do đó phép chia đó **không** so sánh được với bài báo gốc và chỉ được báo cáo như một phép thăm dò độ bền.
5. Lần đánh giá test đóng băng duy nhất (§6).

### 8.6 Ngưỡng độ tin cậy có phải là "lách luật độ đo" không?

Đây là chất vấn công bằng nhất đối với kết quả này, nên chúng tôi trả lời thẳng thay vì chôn nó xuống cuối.

τ, `min_len`, `merge_gap` là **ba số vô hướng** được khớp trên dev và đóng băng trước test — cùng loại lựa chọn như learning rate hay tiêu chí chọn checkpoint. Bằng chứng cho thấy đó không phải overfit dev:

- điểm tối ưu trên dev là một **vùng bằng phẳng rộng**, không phải đỉnh nhọn (§6.3);
- luật được chọn để tối đa **trung bình qua các seed**, không theo seed nào riêng (§5.2);
- độ lệch chuẩn giữa các seed tại điểm được chọn chỉ **0.26**;
- và **kết quả test (50.48) cao hơn dev (47.14)** — ngược hẳn với biểu hiện của overfit dev.

Điều **sẽ là** bất hợp lệ và **đã không** được làm: chỉnh τ trên test, chọn checkpoint trên test, hoặc báo cáo kết quả tốt nhất trong nhiều lần chạy test.

Nhưng có một **hạn chế thật sự** cần nêu: OneIE được báo cáo ở điểm vận hành tự nhiên của nó, còn chúng tôi báo cáo ở điểm vận hành đã tinh chỉnh. Một so sánh ghép cặp precision/recall sẽ thu hẹp khoảng cách Arg-C. Bằng chứng: **recall Arg-I của chúng tôi thực tế thấp hơn OneIE** (53.35 so với 56.29) — chính vì lý do này.

---

## 9. Phân tích lỗi

### 9.1 Phân loại lỗi trên test

Bảng 15 — Phân loại lỗi trên 533 luận cứ vàng được chấm điểm (test, seed 13):

| Loại | Số lượng | Tỉ lệ trên nhãn vàng |
|---|---|---|
| đúng | 246 | 46.2% |
| **bỏ sót hoàn toàn** | **177** | **33.2%** |
| lỗi biên (vai trò đúng, IoU ≤ 0.5) | 44 | 8.3% |
| nhầm vai trò (span đúng, vai trò sai) | 41 | 7.7% |
| chết vì sai loại sự kiện cửa sổ | 25 | 4.7% |
| span thừa | 102 | — |

Độ chính xác phân loại loại sự kiện trên test: **91.41% (149/163)** — tốt hơn trên dev (89.24%), nên chi phí do sai loại giảm còn 4.7%.

### 9.2 Điểm yếu lớn nhất: span ngắn

Bảng 16 — Recall theo độ dài span vàng (test, seed 13):

| Độ dài span | n | recall @ IoU > 0.5 |
|---|---|---|
| **1–4 token** | 77 | **14.3** |
| 5–9 | 141 | 34.8 |
| 10–19 | 202 | 58.4 |
| 20–34 | 106 | 61.3 |
| 35+ | 7 | 42.9 |

Thoạt nhìn đây có vẻ là điểm yếu **do chính chúng tôi gây ra**: luật `min_len = 3` xoá thẳng mọi span ngắn. Hướng khắc phục hiển nhiên là **ngưỡng phụ thuộc độ dài** — dùng τ cao cho span ngắn thay vì cắt cứng. **Chúng tôi đã cài và thử nghiệm nó, và nó thất bại.**

Họ luật được tìm kiếm (vẫn đúng ba tham số, nên không thêm dung lượng khớp trên dev):

```
τ_eff(len) = τ_short   nếu len <  short_len
           = τ_long    nếu len >= short_len
```

Lưới: `τ_long ∈ {0.80…0.92}`, `τ_short ∈ {0.90…0.99}`, `short_len ∈ {3,4,5}`, chọn theo trung bình dev Arg-C IoU qua 3 seed. **Kết quả: +0.04** (47.18 ± 0.21 so với luật đang dùng 47.14 ± 0.26). Bộ tinh chỉnh tự chọn `τ_short = 0.99`, tức nó **chủ động tiếp tục loại bỏ span ngắn**. Luật đang dùng giữ nguyên, nên không phải chấm lại test.

**Vì sao nó thất bại — cơ chế đo được** (dev, seed 42). Các span do head vai trò phát ra, phân theo độ dài *dự đoán*:

| độ dài dự đoán | #đúng | #sai | precision | conf trung vị (đúng) | conf trung vị (sai) |
|---|---|---|---|---|---|
| **1–2** | **7** | **225** | **3.0%** | 0.910 | 0.632 |
| 3–4 | 15 | 89 | 14.4% | 0.963 | 0.819 |
| 5–9 | 55 | 130 | 29.7% | 0.967 | 0.911 |
| 10+ | 179 | 95 | **65.3%** | 0.997 | 0.972 |

Và độ tin cậy **không mang tín hiệu sử dụng được** cho span ngắn: trong các span dự đoán dài < 3 token có `conf ≥ 0.95`, **0 trên 5 là đúng**.

**Diễn giải.** Các đầu ra ngắn của head vai trò không phải là dự đoán ngắn có chủ ý — chúng là **mảnh vỡ** của một phép phân đoạn mức mệnh đề. Head được huấn luyện trên mục tiêu 10–15 token, nên khi nó phát ra một span 1–2 token thì đó là **thất bại**, không phải một quyết định tinh tế. Do đó `min_len = 3` là luật **đúng**, và điểm yếu ở span ngắn là một thuộc tính **biểu diễn**, không phải hiện vật của phép hiệu chỉnh.

Một chỉ số củng cố: **độ lệch độ dài trên các span khớp đúng có trung bình −1.00 token và trung vị +0.0** — biên span không thiên lệch. Mô hình định vị biên tốt; nó chỉ không biết *nên* phát ra span ngắn ở đâu.

### 9.3 Khoảng cách giữa các lĩnh vực

Bảng 17 — Kết quả theo lĩnh vực (test, trung bình ± độ lệch chuẩn qua 3 seed):

| Lĩnh vực | Arg-I IoU F1 | Arg-C IoU F1 | số luận cứ vàng |
|---|---|---|---|
| bioinfo (sinh tin học) | 78.14 ± 2.69 | **74.39 ± 1.88** | 52 |
| ACL (xử lý ngôn ngữ) | 66.24 ± 5.96 | 59.21 ± 4.79 | 60 |
| jmir (tin học y tế) | 57.50 ± 1.34 | 52.11 ± 2.01 | 218 |
| cscw (tính toán xã hội) | 60.16 ± 6.12 | 48.61 ± 2.98 | 91 |
| **dh (nhân văn số)** | 40.14 ± 1.65 | **29.75 ± 1.11** | 112 |

Digital Humanities kém hơn sinh tin học **2.5 lần**, và ổn định qua các seed (độ lệch chuẩn chỉ 1.11). Đây chính là khoảng cách mà bài báo gốc đã chỉ ra, và **phương pháp của chúng tôi không thu hẹp được nó**. Chúng tôi coi đây là một kết quả âm tính cần nêu rõ, không phải một chi tiết để giấu.

Giả thuyết giải thích: văn bản nhân văn số mang tính tường thuật, ranh giới mệnh đề mờ hơn, và các vai trò như Context/Analysis khó tách bạch hơn so với văn bản sinh tin học vốn có cấu trúc rất khuôn mẫu.

### 9.4 Kết quả theo vai trò

Bảng 18 — Kết quả theo từng vai trò (test, trung bình 3 seed, Arg-C IoU):

| Vai trò | P | R | F1 | số mẫu vàng |
|---|---|---|---|---|
| Challenge | 68.19 | 56.32 | **61.67** | 58 |
| Method | 64.69 | 58.97 | **61.63** | 143 |
| Results | 60.71 | 62.36 | **61.50** | 116 |
| Purpose | 56.52 | 30.23 | 39.39 | 43 |
| Implications | 58.42 | 28.07 | 37.28 | 19 |
| **Context** | 34.14 | 28.17 | **30.86** | 142 |
| Analysis | 0.00 | 0.00 | **0.00** | 10 |
| Contradictions | 0.00 | 0.00 | 0.00 | 1 |
| Ethical | 0.00 | 0.00 | 0.00 | 1 |

Hai điều đáng chú ý:

1. **Context vừa là vai trò phổ biến thứ hai (142 mẫu vàng) vừa thuộc nhóm yếu nhất (30.86).** Đây là điểm cải thiện có giá trị cao nhất trong nhóm vai trò, vì nó có cả khối lượng lẫn khoảng dư. Nhìn vào các nhầm lẫn phổ biến nhất (Method→Context, Context→Challenge, Context→Method), có vẻ Context là "vai trò mặc định" mà mô hình gán khi không chắc chắn.

2. **Ba vai trò đuôi đều bằng 0 tuyệt đối.** Với Contradictions (2 mẫu train) và Ethical (1 mẫu train), không mô hình học có giám sát nào làm được gì. Analysis có 67 mẫu train nhưng vẫn về 0 — đây là thất bại thực sự chứ không chỉ là thiếu dữ liệu cực đoan.

---

### 9.5 Thí nghiệm kiểm soát: độ dài span có phải nguyên nhân gây khó không?

Luận điểm của bài dựa trên hình học của mục tiêu. Kiểm chứng nó thông thường cần một benchmark thứ hai — nhưng làm vậy sẽ thay đổi cùng lúc dữ liệu, tập nhãn, bộ chấm và mô hình.

SciEvent cho phép một phép kiểm chặt hơn. **Cùng một cửa sổ chứa hai nhóm mục tiêu**, được trích xuất bởi **cùng một mô hình** trong **cùng một lượt forward**: 9 vai trò được chấm (mệnh đề, 10.9–18.5 token) và Agent/PrimaryObject/SecondaryObject (mention ngắn, 2.4–5.9 token). Mọi thứ khác giữ nguyên: encoder, dữ liệu huấn luyện, tối ưu hoá, giải mã. So khớp dùng chính hàm `iou_overlap` của benchmark.

Bảng 19 — Kiểm soát hình học (test, trung bình 3 seed):

| mục tiêu | nhóm | độ dài TB | gold | P | R | F1 |
|---|---|---|---|---|---|---|
| **Agent** | mention ngắn | **2.4** | 163 | 79.26 | 82.82 | **81.00** |
| PrimaryObject | mention ngắn | 5.9 | 153 | 53.33 | 63.18 | 57.81 |
| Purpose | mệnh đề | 10.9 | 43 | 57.97 | 31.01 | 40.40 |
| **Context** | mệnh đề | **11.0** | 142 | 37.61 | 30.99 | **33.97** |
| Method | mệnh đề | 13.2 | 143 | 65.45 | 59.67 | 62.36 |
| Results | mệnh đề | 13.3 | 116 | 63.49 | 65.23 | 64.32 |
| Challenge | mệnh đề | 14.6 | 58 | 70.28 | 58.05 | 63.56 |
| | | | | | | |
| **mention ngắn (gộp)** | | | 327 | | 70.85 | **67.43** |
| **mệnh đề (gộp)** | | | 531 | | 48.59 | **51.65** |

**Kết quả này bác bỏ giả thuyết hình học ở dạng tổng quát mà chúng tôi từng phát biểu.** Nhóm mention ngắn lại **cao hơn** nhóm mệnh đề (67.43 so với 51.65), và mục tiêu ngắn nhất — Agent, 2.4 token — lại là mục tiêu **tốt nhất** (81.00), trong khi Context ở 11.0 token nằm trong nhóm tệ nhất (33.97).

Ba hệ quả, và chúng tôi ghi lại vì chúng định hình lại tuyên bố của bài:

1. **Độ dài span không quyết định độ khó. Ngữ nghĩa của vai trò mới quyết định.** Context khó không phải vì dài mà vì ranh giới ngữ nghĩa của nó với Method và Results là mờ.
2. **Thiết kế hai head được chứng thực** — và đây chính là bằng chứng mức ablation cho nó: mỗi head xử lý tốt đúng hình học mà nó được huấn luyện.
3. **Tuyên bố của bài phải thu hẹp** về đúng điều có bằng chứng: các luận cứ *được chấm điểm* của SciEvent là mệnh đề nối đuôi nhau, nên một cách đặt bài toán theo phân đoạn thắng bộ máy trích xuất mention **trên chỉ số này**. Chúng tôi **không** tuyên bố điều tổng quát "hình học quyết định họ mô hình", vì thí nghiệm kiểm soát ở trên không ủng hộ nó.

## 10. Hạn chế

Xếp theo mức độ ảnh hưởng đo được:

1. **Span ngắn.** Recall 14.3% ở nhóm 1–4 token và 34.8% ở nhóm 5–9, so với ≈60% ở nhóm 10–34. Do chính luật `min_len = 3` và τ = 0.90 gây ra. Một ngưỡng phụ thuộc độ dài nhiều khả năng lấy lại được phần lớn.

2. **Digital Humanities.** 29.75 so với 74.39 của sinh tin học. Chúng tôi tái hiện khoảng cách này nhưng không thu hẹp nó.

3. **Các vai trò hiếm bằng 0.** Analysis, Contradictions, Ethical.

4. **Conclusions/Implications là loại sự kiện yếu nhất** (34.11 Arg-C IoU), và vai trò Implications cũng yếu (37.28) — hai điều này liên quan với nhau.

5. **Phân loại loại sự kiện vẫn tốn 4.7%** số luận cứ vàng, ở mức chính xác 91.41%.

6. **Khớp chính xác vẫn thấp** (Arg-C EM 33.43 so với IoU 50.48). Mô hình lấy span *xấp xỉ* đúng, không *chính xác* đúng. Đủ cho chỉ số chính của benchmark này, không đủ cho một benchmark khắt khe hơn.

7. **Điểm vận hành thiên precision.** Recall Arg-I thấp hơn OneIE 2.94 điểm.

8. **Không mô hình hoá bước phân đoạn.** Chúng tôi tiêu thụ các cửa sổ vàng, đúng như ba baseline được tinh chỉnh. Chúng tôi không báo cáo con số phân đoạn nào, thay vì báo cáo một con số không so sánh được.

### 10.1 Một đặc trưng bị cố ý bỏ không dùng

Mã định danh cửa sổ mã hoá **vị trí thứ tự của đoạn trong tóm tắt**. Một luật đa số đơn giản chỉ dựa trên thứ tự này dự đoán đúng loại sự kiện với độ chính xác **98.73%** trên dev, so với **89.24%** (dev) / **91.41%** (test) của mô hình dựa trên văn bản:

| Vị trí đoạn | Loại chiếm đa số | Tỉ lệ | n (train) |
|---|---|---|---|
| 0 | Background/Introduction | 97.5% | 400 |
| 1 | Methods/Approach | 95.7% | 394 |
| 2 | Results/Findings | 88.5% | 330 |
| 3 | Conclusions/Implications | 100.0% | 154 |

Kết hợp với Bảng 3 (oracle loại sự kiện đáng giá +4.72 Arg-C), phần lớn khoảng dư này là **có thể lấy được thật**.

Chúng tôi **cố ý loại nó khỏi hệ thống chính**, vì các baseline trong bài báo gốc không dùng nó, và mục tiêu của chúng tôi là một so sánh ngang hàng. Nó chỉ được báo cáo như một khoảng dư đã đo. Lưu ý rằng một pipeline triển khai thực tế **hoàn toàn có quyền** dùng đặc trưng này — bất kỳ hệ thống nào phân đoạn một tóm tắt đều biết đoạn nào đứng trước đoạn nào.

---

## 11. Kết luận

Chúng tôi bắt đầu bằng một câu hỏi đơn giản: *giả định ngầm của các baseline — rằng luận cứ là những mention thực thể ngắn — có đúng với SciEvent không?* Câu trả lời, đo được từ dữ liệu, là không. Luận cứ được chấm điểm của SciEvent là những **mệnh đề dài 10–15 token, nối đuôi nhau, gần như không chồng lấn**, trong các đơn vị chứa đúng một sự kiện.

Xây dựng theo đúng hình học đó — một bộ gán nhãn span mức từ với hai đầu BIO tách rời và điều kiện hoá theo loại sự kiện — cộng với một luật giải mã ba tham số hiệu chỉnh trên dev, cho **Arg-C IoU 50.38, khoảng tin cậy bootstrap 95 % [45.49, 55.48]** trên test. Cận dưới của khoảng này vẫn cao hơn kết quả tốt nhất của bài báo gốc 3.88 điểm, và **100 %** số lần lấy mẫu lại đều vượt — đây là tuyên bố vững chắc của bài. Arg-I (57.88) cải thiện nhưng khoảng tin cậy chứa baseline; trigger ROUGE-L (76.96) tương đương chứ không vượt. Chúng tôi không tuyên bố ba chỉ số như nhau.

Ba điều chúng tôi cho là đáng mang đi nhất:

1. **Hãy đo dữ liệu trước khi chọn kiến trúc.** Độ dài span, mức chồng lấn, số sự kiện mỗi đơn vị, khoảng cách giữa các span. Chúng tôi làm việc này *trước* khi đọc literature EAE, và chính nó ngăn chúng tôi đi vào nhánh sinh văn bản mà DEGREE đã chứng minh là thất bại ở đây. Nhưng phép đo phải được kiểm chứng: thí nghiệm kiểm soát của chính chúng tôi (§9.5) **bác bỏ** phiên bản tổng quát của luận điểm này — độ dài span không quyết định độ khó, ngữ nghĩa vai trò mới quyết định. Điều còn đứng vững là hẹp hơn và cụ thể hơn: luận cứ *được chấm điểm* của SciEvent là mệnh đề nối đuôi, nên phân đoạn là cách đặt bài toán đúng cho *chỉ số này*.

2. **Việc đặt lại bài toán mua recall, không mua F1.** Bảng 8 cho thấy nếu không hiệu chỉnh, chúng tôi *thua* OneIE. Điều thực sự thay đổi là recall (+8.12). Hiệu chỉnh là thứ biến khoảng dư đó thành F1 — và nó không thể làm vậy nếu khoảng dư không có sẵn.

3. **Suy ra kiến trúc từ thống kê dữ liệu là cám dỗ — và ở đây nó sai hai lần.** Chúng tôi đo đúng rằng chồng lấn giữa nhóm là 5.9 % và rằng độ đo nhạy với loại sự kiện, rồi rút ra hai quyết định kiến trúc từ đó. Ablation đơn-yếu-tố (§7.6) bác bỏ **cả hai**: việc tách hai head và đường điều kiện hoá loại sự kiện đều **trơ**. Phép đo đúng không bảo đảm suy luận đúng; **chỉ ablation mới phân xử được**. Sáu thành phần bị bác bỏ tổng cộng, và trong bốn trường hợp chúng tôi đo được *vì sao* dự đoán ban đầu sai — CRF thất bại vì cắt vụn chưa bao giờ là lỗi chủ đạo (bằng chứng độc lập: bộ tinh chỉnh tự chọn `merge_gap = 0`); head vai trò mức span thất bại vì ngưỡng đã xoá sẵn phần lớn lỗi nó nhắm tới; ngưỡng theo độ dài thất bại vì span ngắn là mảnh vỡ chứ không phải dự đoán có chủ ý; và giả thuyết hình học tổng quát bị chính thí nghiệm kiểm soát của chúng tôi bác bỏ.

### 11.1 Hướng tiếp theo

1. **Ngưỡng phụ thuộc độ dài** — nhắm thẳng vào điểm yếu lớn nhất (recall 14.3% ở span ngắn), chi phí gần bằng 0, và sẽ làm yếu đi chất vấn "phần lớn biên độ đến từ ngưỡng".
2. **Xử lý riêng cho Digital Humanities** — thử ngưỡng theo lĩnh vực trước (rẻ), rồi tới adapter nếu khoảng cách mang tính biểu diễn chứ không phải hiệu chỉnh.
3. **So sánh backbone đúng nghĩa** — ModernBERT-large đã chuẩn bị nhưng chưa chạy.
4. **Cải thiện vai trò Context** — vai trò có khối lượng lớn nhất trong nhóm yếu.
5. **Ensemble tương thích với hiệu chỉnh** — ví dụ bỏ phiếu trên các span đã giải mã, hoặc hiệu chỉnh từng mô hình trước khi lấy trung bình, thay vì trung bình xác suất hậu nghiệm thô.
6. **Kiểm chứng luận điểm về hình học trên benchmark khác** — cụ thể là chạy trên một benchmark có span **ngắn** (ví dụ RAMS) như một **đối chứng âm** có chủ đích: nếu cách đặt lại bài toán của chúng tôi *thất bại* ở đó đúng như hình học dự đoán, thì luận điểm trung tâm được củng cố chứ không bị suy yếu.

---

## Phụ lục A. Tái lập

```bash
cd CARVE
pip install -r requirements.txt

bash scripts/reproduce.sh      # kiểm tra → 3 seed → đóng băng trên dev → test một lần → phân tích
bash scripts/audit.sh          # kiểm toán rò rỉ và giao thức (§8)
python3 scripts/results_tables.py # sinh lại các bảng 5, 6, 7, 17, 18
```

Tổng chi phí tính toán cho toàn bộ công trình: **≈2.5 giờ GPU** trên một card NVIDIA RTX 5880 Ada (49 GB).

Các tệp quan trọng:

| Tệp | Nội dung |
|---|---|
| `assets/decoding_rules.json` | luật giải mã đã đóng băng (τ = 0.90, min_len = 3) |
| `preds/test_preds_s{42,13,101}.jsonl` | dự đoán test đã đóng băng |
| `preds/docsplit_test_s{42,13,101}.jsonl` | dự đoán trên phép chia rời rạc theo tài liệu |
| `runs/<run>/log.json` | lịch sử từng epoch của mọi lần chạy |
| `PAPER_NOTES.md` | bản ghi đầy đủ mọi con số đã đo, kèm cấu hình chính xác |

## Phụ lục B. Tóm tắt các quyết định thiết kế và bằng chứng tương ứng

| Quyết định | Bằng chứng buộc phải làm vậy |
|---|---|
| Gán nhãn BIO thay vì liệt kê span | Trần của cách biểu diễn đo được là 99.70 Arg-C IoU — chỉ tốn 0.3 F1 |
| Hai đầu BIO tách rời | Chồng lấn giữa hai nhóm là 5.9%, trong khi trong nhóm chỉ ≤ 2.4% |
| Span ngắn thắng khi xung đột BIO | Khôi phục được 98.97% span vàng chính xác |
| Đầu loại sự kiện tường minh + điều kiện hoá | Oracle loại sự kiện đáng giá +4.72 Arg-C |
| Scheduled sampling cho điều kiện hoá | Khớp phân phối giữa huấn luyện và suy luận |
| Giải mã có ngưỡng | 784 span dự đoán so với 497 span vàng — lỗi hiệu chỉnh |
| Luật toàn cục 3 tham số, không theo vai trò | Ngưỡng theo vai trò chỉ thêm +0.67 với 9 tham số trên 158 mẫu dev |
| Một mô hình đơn, không ensemble | Ensemble làm tệ đi giải mã có hiệu chỉnh (47.10 so với 48.10) |
| Không dùng CRF | Bộ tinh chỉnh tự chọn `merge_gap = 0`; CRF chậm hơn 1.7 lần và thấp hơn |
| Trọng số chủ fp32 | Checkpoint fp16 + AdamW → NaN ngay bước tối ưu đầu tiên |
| Không dùng thứ tự đoạn | Các baseline không dùng; giữ so sánh ngang hàng |
