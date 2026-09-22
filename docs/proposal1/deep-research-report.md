# SciEvent dưới ràng buộc ≤30 GB VRAM: audit benchmark, trạng thái SOTA và thiết kế phương pháp TARS-SciEvent

## Kết luận điều hành và trạng thái của tuyên bố SOTA

**Kết luận quan trọng nhất:** tại thời điểm nghiên cứu này, **chưa có cơ sở để tuyên bố đã đạt SOTA mới trên SciEvent**. Tôi đã có thể tái dựng benchmark, audit phần quan trọng của code chính thức, reverse-engineer hợp đồng metric cho extraction, kiểm tra logic split, xác minh các baseline được công bố, tìm kiếm các công trình hậu SciEvent đến ngày **22/09/2026**, và thiết kế một hướng mô hình có xác suất cạnh tranh cao dưới 30 GB. Nhưng môi trường hiện tại không có GPU thích hợp để chạy huấn luyện SciEvent và không có đầy đủ raw CSCW text cần cho một số bước tái tạo dữ liệu; vì vậy các điều kiện “reproduce baseline”, “đo peak CUDA VRAM”, “multi-seed test”, “ablation”, “statistical significance” của SOTA gate **chưa được thỏa mãn**.

SciEvent là benchmark EMNLP 2025 của Dong et al., gồm **500 abstract khoa học**, năm domain và bốn loại event. Paper báo cáo **4.929 event** và **8.911 structured mentions**, với trung bình **2,95 câu/event**. citeturn9search17turn15view0 Trong ba tuning-based baseline của paper, OneIE là mô hình mạnh nhất: Trigger ROUGE-L F1 **72,40**, Argument Identification IoU F1 **53,57**, Argument Classification IoU F1 **41,61**; DEGREE và EEQA thấp hơn đáng kể ở Arg-C. citeturn12view2turn12view3turn15view2

Việc tìm kiếm các công trình từ sau SciEvent tới **22/09/2026** không phát hiện một công trình hậu 2025 nào mà tôi có thể xác minh là báo cáo kết quả tuning-based trực tiếp, cùng split và cùng evaluator trên **benchmark 500-document của Dong et al.**. Do đó, **41,61 Arg-C IoU F1 của OneIE là mốc tốt nhất được công bố và xác minh được trong protocol mà paper mô tả**, chứ chưa nên gọi là “SOTA tuyệt đối hiện tại” do một bất nhất nghiêm trọng giữa mô tả split trong paper và `split_data.py`. Một nguồn dễ gây nhầm là công trình ACL 2026 về **SciEvents/EXCEEDS**: đó là một dataset complex-event khoa học khác, với quy mô và task khác, không phải SciEvent của Dong et al.; các con số của nó không được phép đưa vào bảng so sánh trực tiếp này. citeturn9search23turn9search29

### Phán quyết nghiên cứu

| Hạng mục | Kết luận hiện tại |
|---|---|
| Best published tuning-based reference | **OneIE: Arg-C IoU F1 41,61** |
| Certified current SOTA | **Chưa thể chứng nhận** |
| Lý do chính | Split paper/code chưa được chứng minh tương đương; chưa rerun baseline; chưa có multi-seed |
| Tuning regime nên ưu tiên | Full fine-tuning encoder khoảng 300–400M + structured span-set extraction |
| Phương pháp đề xuất | **TARS-SciEvent — Tuple-Anchored Rhetorical Span-set Extraction** |
| Novelty cốt lõi | Agent–Action–Object trigger tuple → role-conditioned span-set extraction + codebook prototypes |
| Mục tiêu VRAM | Giữ observed peak ≤ **28 GiB**, để có >2 GiB safety margin so với giới hạn 30 GB |
| Backbone ưu tiên thử trước | Modern bidirectional encoder ~300–400M; SciBERT là scientific-domain control |
| PEFT/QLoRA 3B–7B | Nhánh phụ, báo cáo riêng; không trộn với full-FT |
| External scientific DAPT | Nhánh riêng vì thay đổi data regime |
| SOTA claim hiện tại | **Không đạt SOTA gate; “candidate method pending empirical verification”** |

**Evidence:** OneIE vượt hai tuning baselines còn lại ở cả Trigger và Arg-C; paper cũng cho thấy khoảng cách domain/role rất lớn. citeturn12view2turn12view3turn12view4turn13view0

**Interpretation:** cơ hội cải thiện lớn không nằm ở “scale model” đơn thuần mà ở ba nơi: **boundary modeling**, **role discrimination**, và **cross-domain/discourse structure**.

**Confidence:** cao đối với chẩn đoán metric; trung bình đối với lựa chọn architecture cho tới khi có dev experiments.

**Thí nghiệm có thể bác bỏ:** một span-set head với cùng backbone mà không cải thiện đáng kể Arg-I/Arg-C EM so với BIO/OneIE-style head sẽ bác bỏ giả thuyết rằng boundary representation là nút thắt kiến trúc chính.

## Tái dựng SciEvent và hợp đồng đánh giá

### Dataset và schema

Paper mô tả tập dữ liệu gồm năm domain, thu thập 500 abstract khoa học. Việc lấy mẫu yêu cầu abstract có ít nhất ba câu và ít nhất hai event có thể nhận diện. citeturn12view0turn15view0

| Domain | Nguồn/năm trong paper | Số document |
|---|---|---:|
| NLP | ACL 2023 | 100 |
| Social Computing | CSCW 2023 | 100 |
| Computational Biology / Bioinformatics | 2023 | 100 |
| Digital Humanities | DHQ 2021–2023 | 120 |
| Medical Informatics | JMIR 2023 | 80 |
| **Tổng** |  | **500** |

Paper định nghĩa bốn event type. `Background` bao phủ bối cảnh, động cơ, vấn đề, gap, objective và research question; `Method` bao gồm methodology, framework, algorithm, dataset, setup hoặc tool; `Result` biểu diễn observation/output/trend/discovery; `Conclusion` mô tả ý nghĩa, impact, application, vấn đề được giải quyết, recommendation và future work. citeturn15view1

Trigger của SciEvent khác trigger verb thông thường của ACE: nó có cấu trúc

\[
T=(Agent,\ Action,\ PrimaryObject,\ SecondaryObject?)
\]

trong đó Action thường là verb/verb phrase; Agent là thực thể khởi tạo/thực hiện; Object là đối tượng chịu tác động hoặc focus của hành động. Khi object gồm hai span không liên tục, annotation tách thành `Primary Object` và `Secondary Object`. citeturn12view0turn15view1

Một chi tiết annotation đáng lưu ý: guideline có quy tắc riêng cho cấu trúc bị động, nên ý nghĩa annotation của `Agent`/`Object` phải được học theo **codebook của SciEvent**, không nên áp đặt semantic-role convention từ PropBank hoặc ACE. Guideline cũng có priority rule khi một span có vẻ phù hợp nhiều role. citeturn13view1turn13view2

Chín semantic argument roles là: citeturn15view1

| Role | Diễn giải theo codebook | Count công bố |
|---|---|---:|
| Context | nền tảng/tình huống mà event xảy ra | 1.352 |
| Purpose | mục đích/aim | 323 |
| Method | kỹ thuật, tool, methodology, framework | 1.219 |
| Result | observation/output phát sinh | 1.143 |
| Analysis | interpretation/explanation của kết quả | 90 |
| Challenge | constraint, difficulty, weakness | 511 |
| Ethical | ethical issue/concern/justification | **3** |
| Implication | significance/application/future implication | 285 |
| Contradiction | disagreement với existing knowledge | **3** |

Event-type distribution là `Background=1.373`, `Method=1.795`, `Result=1.241`, `Conclusion=520`, tổng cộng **4.929 events**. citeturn12view1

Điều này cho thấy imbalance cực mạnh: Ethical và Contradiction chỉ có ba annotation được báo cáo cho mỗi role, trong khi Context có 1.352. Với tần suất như vậy, việc tối ưu macro rare-role F1 bằng một classifier thuần supervised sẽ có variance rất lớn. Đây là một lý do trực tiếp cho đề xuất role-definition prototype ở phần sau. citeturn12view1

Một số thống kê mà prompt yêu cầu **không được paper công bố dưới dạng đủ để tái dựng**, và tôi không xem chúng là đã biết:

- phân phối document length;
- phân phối event/span length đầy đủ;
- phần trăm event thật sự multi-sentence, thay vì chỉ biết mean = 2,95 sentence/event;
- tỷ lệ trigger có Secondary Object/discontinuous object;
- tỷ lệ event có nhiều argument cùng role;
- lexical seen/unseen distributions.

Những con số này phải được tính trực tiếp trên JSON release sau khi canonicalization, không nên suy từ ví dụ hoặc figure. Việc paper báo trung bình 2,95 câu/event không cho phép suy ra phần trăm event multi-sentence. citeturn15view0

### Task dependency và gold/predicted information

Task nên được hiểu theo graph:

```text
Document
   │
   ▼
Event segmentation ──► Event type
   │                    │
   └──────────┬─────────┘
              ▼
 Agent + Action + PrimaryObject (+ SecondaryObject)
              │
              ▼
       Candidate arguments
              │
              ▼
         Role classification
```

Paper mô tả Task 1 là segmentation/type identification; Task 2 là trigger extraction khi event span/type đã được nhận diện; Task 3 là argument extraction. citeturn15view0turn14view3 Tuy nhiên, **published fine-tuning experiment không phải một end-to-end document → events → triggers → arguments pipeline thống nhất**.

Audit code cho thấy extraction instances được xây dựng quanh **gold event windows/segments**, sau đó từng baseline xử lý type/trigger khác nhau:

| Baseline | Event window | Event type / trigger behavior | Argument dependency |
|---|---|---|---|
| OneIE | gold extraction window | joint structured prediction trong window | event/role graph decoding |
| DEGREE | gold window | generation conditioned bằng event/template | arguments generated theo template |
| EEQA | gold window | trigger QA trước | argument QA chạy trên **predicted trigger file** |

Đặc biệt, README chính thức hướng dẫn EEQA tạo `trigger_predictions.json` rồi dùng file này làm input cho argument QA, trong khi gold EEQA file chỉ đóng vai trò ground truth đánh giá. Vì vậy, gọi mọi extraction score là “gold trigger setting” sẽ là sai.

### Hợp đồng metric extraction đã reverse-engineer

Paper định nghĩa Argument Identification (`Arg-I`) và Argument Classification (`Arg-C`) với Exact Match và IoU; IoU match yêu cầu overlap **lớn hơn 0,5**, không phải ≥0,5. citeturn15view0turn14view4

Audit trực tiếp `baselines/ONEIE/EM_overlap_eval.py` cho thấy contract thực tế chi tiết hơn:

\[
IoU(P,G)=
\frac{|P\cap G|}{|P\cup G|}
\]

và một candidate chỉ match nếu

\[
IoU(P,G) > 0.5.
\]

Argument metrics là **micro P/R/F1**. Matching là **greedy one-to-one**, không phải maximum-weight bipartite matching. Mỗi gold span chỉ được dùng một lần; duplicate predictions còn lại trở thành false positives.

`Arg-I` yêu cầu:

\[
eventType_p=eventType_g
\quad\land\quad
spanMatch(p,g)
\]

còn `Arg-C` thêm:

\[
role_p=role_g.
\]

Một phát hiện đặc biệt quan trọng từ evaluator: **argument score không yêu cầu trigger span/text đúng**; nó yêu cầu event type và argument span/role. Nói cách khác, trigger có thể sai nhưng arguments trong cùng event instance vẫn nhận credit nếu event type và arguments match. Đây là lý do phải tách “metric score” khỏi “end-to-end semantic correctness”.

Agent, Primary Object và Secondary Object được xem là trigger components và bị loại khỏi semantic argument F1.

Trigger metric được tạo bằng cách nối các thành phần trigger đã tìm được và tính ROUGE-L bằng `rouge_score.RougeScorer(..., use_stemmer=True)`, sau đó lấy trung bình theo instance — về bản chất là macro instance-level ROUGE-L.

Một điểm đáng lo khác: evaluator OneIE iterate trên IDs có trong gold. Một prediction có `sent_id` hoàn toàn không tồn tại trong gold có thể **không bị tính false positive**. Đây là evaluator vulnerability; không được phép lợi dụng nó. Headline result phải chạy evaluator chính thức để tương thích, đồng thời nên báo thêm corrected score từ wrapper kiểm tra ID coverage.

### Synthetic unit tests của metric

Tôi đã kiểm tra logic contract nói trên bằng các trường hợp synthetic tương ứng với code evaluator; đây là **verification của evaluator logic**, không phải kết quả trên SciEvent test set.

| Gold / Prediction | Official-contract outcome |
|---|---|
| identical span | match |
| Gold `[2,4)`, pred `[2,3)` | IoU = 0,5 → **không match** |
| Gold `[2,4)`, pred `[1,4)` | IoU = 2/3 → match |
| Gold `[2,4)`, pred `[1,5)` | IoU = 0,5 → **không match** |
| 2 identical predictions / 1 gold | P=.5, R=1, F1=.667 |
| 1 correct pred / 2 gold | P=1, R=.5, F1=.667 |
| đúng span nhưng sai role, Arg-C | F1=0 |
| empty pred + empty gold | implementation trả 0 |
| prediction trên ID ngoài gold | có thể bị bỏ qua |

Có thêm một hệ quả: nếu một prediction có thể overlap >0,5 với nhiều gold spans, lựa chọn gold đầu tiên theo iteration order có thể ảnh hưởng kết quả. Vì vậy official evaluator không tương đương với global-optimal matching.

**Evidence:** contract trên đến từ evaluator code chính thức và threshold trong paper. citeturn15view0turn14view4

**Interpretation:** boundary modeling rất quan trọng, nhưng tối ưu trực tiếp “vừa qua 0,5” là metric gaming. Mô hình mới nên tối ưu span correctness thực sự và báo cả EM lẫn IoU.

**Confidence:** cao.

**Falsifier:** wrapper chạy cùng official predictions nhưng cho score khác official evaluator ở các test hợp lệ sẽ cho thấy reconstruction metric chưa chính xác; mọi experiment phải dừng cho tới khi sai lệch được giải quyết.

## Audit repository, split, leakage và reproducibility

Repo chính thức được audit trên branch `EMNLP-2025`, đặc biệt các path mà prompt yêu cầu: `README.md`, `SciEvent_data/annotated/*`, `data_scripts/shared/*`, `baselines/ONEIE/*`, `baselines/DEGREE/*`, và `baselines/EEQA/*`.

### Split: bất nhất nghiêm trọng nhất

**FACT FROM PAPER:** paper nói tuning experiments dùng split **80/10/10 theo document**. citeturn15view0

**FACT FROM CODE:** `data_scripts/shared/split_data.py` không triển khai group split theo `doc_id`. Logic của script là:

```text
for each instance:
    find all event types in instance
    add instance to every corresponding event-type bucket

for each event-type bucket independently:
    deduplicate by wnd_id
    random.shuffle(seed=42)
    split 80 / 10 / remainder

concatenate all train buckets
deduplicate train by wnd_id

same independently for dev/test
```

Nó **không cross-deduplicate train/dev/test**, và không enforce disjointness theo `doc_id`.

Điều này tạo ra hai failure modes.

Thứ nhất, một abstract có nhiều event windows có thể có các window đi sang split khác nhau, làm vi phạm document-level split.

Thứ hai, nghiêm trọng hơn, một **instance có nhiều event types** được đưa vào nhiều event-type buckets; do mỗi bucket được shuffle/split độc lập, cùng `wnd_id` về mặt thuật toán có thể rơi vào train qua một bucket và dev/test qua bucket khác. Dedup sau đó chỉ diễn ra *bên trong* từng split, nên code không ngăn cross-split `wnd_id` overlap.

Đây là **INFERENCE FROM CODE**, chưa phải bằng chứng rằng released split thực tế có overlap khác 0. Muốn kết luận leakage thực nghiệm bắt buộc phải chạy:

```python
train_doc & dev_doc
train_doc & test_doc
dev_doc & test_doc

train_wnd & dev_wnd
train_wnd & test_wnd
dev_wnd & test_wnd
```

trên chính các generated files đã được dùng để tạo result của paper.

Do generated split artifacts không được kiểm tra vào repo theo một cách cho phép tôi tái dựng đầy đủ ngay trong môi trường hiện tại, **tôi không tuyên bố “SciEvent published result definitely leaked”**. Kết luận khoa học chính xác là:

> Paper yêu cầu document-disjoint split, nhưng implementation hiện tại không enforce điều đó và còn cho phép cùng window xuất hiện ở nhiều split. Tính disjoint của split đã sử dụng để tạo bảng paper cần được kiểm chứng thực nghiệm trước khi coi 41,61 là một benchmark score trên strict document split.

Đây là vấn đề ưu tiên số một trước model development.

### CSCW và khả năng tái tạo dữ liệu

README chính thức cho biết raw abstracts được cung cấp cho các nguồn khác, nhưng **CSCW text không được phân phối do licensing**. Vì vậy tái tạo preprocessing từ raw source đòi hỏi lấy hợp pháp CSCW abstracts riêng.

`prepare_segmentation.py` có một hành vi nguy hiểm: khi abstract/segment tương ứng không được tìm thấy, code có thể tạo instance với tokens/pieces/sentence/mentions rỗng thay vì fail-fast. Nếu người dùng chạy pipeline mà thiếu CSCW source, experiment có nguy cơ “thành công” về mặt script nhưng sinh data hỏng.

Bản reproducibility nghiêm túc nên thay hành vi này bằng:

```text
--fail_on_missing_abstract=true
```

và assert:

\[
N_{\text{missing}} = 0
\]

trước khi train.

### OneIE audit

Config chính thức dùng `bert-large-cased`, batch size 2, task LR \(10^{-3}\), BERT LR \(10^{-5}\), tối đa 60 epochs và beam size 20.

Một chi tiết rất dễ bị đọc sai là:

```json
"use_global_features": true,
"global_features": []
```

`[]` **không có nghĩa là tắt mọi global feature**. Trong `global_feature.py`/model logic, điều kiện có dạng tương đương:

```python
if feature in selected_features or not selected_features:
    use(feature)
```

nên empty list thực tế kích hoạt toàn bộ feature families.

Tuy nhiên, hypothesis “OneIE thắng vì hard event-role admissibility constraints” lại không phù hợp với resource SciEvent: `resource/scievent/event_role.json` ánh xạ **mỗi bốn event type tới toàn bộ roles**. Như vậy OneIE không thắng đơn giản vì nó loại bỏ các role bất hợp lệ theo event type. Lợi thế nếu có phải đến từ joint representation, beam/global scoring, feature interactions hoặc preprocessing.

Đây là một kết quả audit quan trọng vì paper giải thích lợi ích OneIE bằng structured/global modeling, nhưng ta không nên biến lời giải thích đó thành causal claim khi chưa ablate. OneIE ban đầu được thiết kế cho joint IE với global features, đúng với architecture của công trình gốc. citeturn19search11

### DEGREE audit

Config SciEvent sử dụng `facebook/bart-large`, batch 4, gradient accumulation 2, LR \(10^{-5}\), tối đa 45 epochs, max input 1024, generation max length 100, beam size 3 và tối đa 15 negative templates.

Có hai dấu hiệu legacy đáng audit nhưng không được tự động coi là bug:

- data-generation path vẫn dùng string `"ace05e"` để dispatch template machinery;
- vocabulary/schema generation code xem cả train/dev/test khi xây label vocabulary.

Trường hợp thứ hai không nhất thiết là instance-label leakage vì ontology SciEvent vốn là fixed schema và public; nhưng reproducibility nghiêm ngặt nên xây vocabulary từ **declared schema** hoặc train split, không cần đọc test annotation.

DEGREE là conditional-generation approach có nguồn gốc từ prompt/template-based event extraction. citeturn19search4

### EEQA audit

EEQA config dùng `bert-base-uncased`, sequence length 512, batch 16, LR \(3\times10^{-5}\), 10 epochs, `n_best_size=5`, `max_answer_length=20` và thresholding tương đối với CLS thông qua `--larger_than_cls`.

README pipeline xác nhận argument stage nhận **trigger predictions**, do đó trigger false negatives trực tiếp triệt tiêu recall của argument QA. Điều này giúp giải thích vì sao pipeline QA có thể chịu error propagation mạnh hơn OneIE. EEQA gốc chính là cách formulate event extraction thành các natural-language questions. citeturn19search0

### Audit verdict

| Vấn đề | Mức độ | Trạng thái |
|---|---|---|
| Paper nói document split nhưng script không group theo doc | **Critical** | xác nhận từ code |
| Same `wnd_id` có thể rơi nhiều split | **Critical** | code cho phép; chưa đo actual overlap |
| CSCW raw text thiếu do license | High | xác nhận |
| Missing abstract có thể sinh empty instance | High | xác nhận từ preprocessing |
| Evaluator bỏ qua prediction IDs ngoài gold | High | xác nhận từ code |
| Greedy/order-sensitive IoU matching | Medium | xác nhận |
| IoU dùng `>0.5`, không phải `>=` | Medium | xác nhận |
| OneIE `global_features=[]` thực ra bật toàn bộ | Medium | xác nhận |
| OneIE hard event-role constraint giải thích gain | **Không được ủng hộ** | all types cho phép all roles |
| DEGREE `"ace05e"` legacy dispatch | Medium | cần ablation/inspection thêm |
| DEGREE schema vocab nhìn test | Medium | tránh trong strict reproduction |
| Test-set threshold selection | Chưa thấy đủ bằng chứng | không khẳng định |
| CSCW khiến published split không tái dựng được từ raw repo đơn thuần | High | cần licensed source |

Do split issue, mọi paper mới nên duy trì **hai protocol hoàn toàn riêng**:

**Official-code protocol:** dùng đúng artifact/split đã tạo kết quả paper, để so sánh lịch sử.

**Strict-document protocol:** group toàn bộ instances cùng `doc_id` trước khi stratification, đảm bảo zero intersections theo `doc_id` và `wnd_id`.

Không được đưa số của hai protocol vào cùng cột “SOTA”.

**Evidence:** source code trực tiếp mâu thuẫn với mô tả document-level split của paper. Paper tuyên bố 80/10/10 by-document. citeturn15view0

**Interpretation:** đây hiện là rủi ro reproducibility lớn hơn bất kỳ lựa chọn architecture nào.

**Confidence:** cao đối với code discrepancy; trung bình đối với việc actual published artifact thực sự leak.

**Falsifier:** nếu exact split artifacts của authors cho `train.doc_id ∩ dev.doc_id = train.doc_id ∩ test.doc_id = dev.doc_id ∩ test.doc_id = ∅` và tương tự cho `wnd_id`, thì leakage concern đối với *published artifact* được bác bỏ, mặc dù script vẫn cần sửa.

## Baselines, current SOTA và error budget

### Published tuning-based reference

Các số dưới đây là **paper results, không phải rerun của nghiên cứu này**. citeturn12view2turn12view3turn15view2

| Model | Backbone | Regime | Trigger RL-F1 | Arg-I EM | Arg-I IoU | Arg-C EM | Arg-C IoU |
|---|---|---|---:|---:|---:|---:|---:|
| EEQA | BERT-base | full tuning | 45,05 | 14,63 | 32,91 | 11,88 | 26,51 |
| DEGREE | BART-large | full tuning | 56,85 | 19,79 | 29,84 | 15,07 | 21,57 |
| **OneIE** | BERT-large-cased | full tuning | **72,40** | **33,57** | **53,57** | **26,61** | **41,61** |

Một GPT few-shot setting trong paper đạt Trigger F1 **75,08**, cao hơn OneIE về trigger, nhưng argument score không vượt OneIE; đây cũng không phải tuning-based regime và không nên nhập chung leaderboard. citeturn12view2turn12view3

Từ literature search tới 22/09/2026:

| SOTA category | Best result xác minh được trên exact Dong et al. SciEvent |
|---|---|
| `SOTA_same_protocol` | OneIE **41,61 Arg-C IoU F1**, với caveat split |
| `SOTA_full_finetune` | OneIE, cùng caveat |
| `SOTA_PEFT` | **Không tìm thấy result trực tiếp có thể xác minh** |
| `SOTA_external_data` | **Không tìm thấy result trực tiếp có thể xác minh** |
| absolute/current certified SOTA | **Chưa đủ điều kiện chứng nhận** |

Điều này khác với nói “không có paper nào tồn tại”: kết luận chính xác chỉ là **không tìm thấy công trình hậu SciEvent có kết quả cùng benchmark/protocol đủ rõ để thay thế OneIE làm published reference**. Công trình ACL 2026 EXCEEDS dùng một SciEvents benchmark khác và phải loại khỏi comparison. citeturn9search23turn9search29

### Tại sao OneIE thắng?

Arg-C IoU precision/recall của ba tuning systems là:

- EEQA: 25,85 / 27,20;
- DEGREE: **48,99 / 13,83**;
- OneIE: 39,69 / **43,71**. citeturn12view3

Điểm nổi bật không phải OneIE có precision cao nhất; **DEGREE mới có precision cao nhất**. OneIE thắng F1 chủ yếu vì recall cao hơn rất nhiều.

Điều đó phù hợp với ba cơ chế khả dĩ:

1. structured joint decoding của OneIE không phụ thuộc vào autoregressive template generation cho mỗi argument;
2. EEQA chịu error propagation từ trigger QA sang argument QA;
3. DEGREE có tendency “conservative generation”: precision cao nhưng bỏ sót nhiều argument.

Tuy nhiên, **chưa thể phân rã causal gain** thành “BERT representation vs global features vs objective vs preprocessing”. Những yếu tố này bị confound trong bảng paper. OneIE gốc được thiết kế như joint structured IE model với global features, còn DEGREE là conditional generation và EEQA là QA-based extraction. citeturn19search11turn19search4turn19search0

Một giả thuyết có thể loại bỏ ngay là “OneIE thắng chủ yếu nhờ hard event-role constraints”: resource SciEvent cho phép mọi role đối với mọi event type.

### Boundary detection so với role classification

Hai chênh lệch metric rất có ý nghĩa:

\[
53.57-33.57=\mathbf{20.00}
\]

điểm F1 giữa Arg-I IoU và Arg-I Exact Match.

Trong khi đó:

\[
53.57-41.61=\mathbf{11.96}
\]

điểm giữa Arg-I IoU và Arg-C IoU. citeturn12view3turn15view2

Không thể diễn giải hai phép trừ này như decomposition chính xác của error probability, nhưng chúng cho thấy **boundary exactness là bottleneck ít nhất ngang, và theo metric gap còn lớn hơn role discrimination**.

Đây là lý do span-set modeling được ưu tiên hơn việc chỉ thay class-balanced loss.

### Domain và role error budget

Figure của paper cho thấy OneIE Arg-C IoU mạnh nhất ở NLP/Computational Biology và yếu đặc biệt ở Digital Humanities; từ figure, các giá trị xấp xỉ vào khoảng NLP ~60, Computational Biology ~53, Social Computing ~43, Medical Informatics ~42, Digital Humanities ~24. Đây là số đọc từ plot nên không nên báo với độ chính xác hai chữ số. citeturn13view0

Role plot tương tự cho thấy Method/Challenge/Result tương đối mạnh, Context và Implication khó hơn, còn Analysis/Ethical/Contradiction cực yếu; Ethical và Contradiction chỉ có ba training-corpus annotations ở quy mô toàn dataset nên kết quả thấp là không bất ngờ. citeturn12view1turn12view4

Event-type plot cho thấy Conclusion yếu hơn, còn Method events mặc dù phổ biến vẫn phức tạp; paper cũng nhận xét event-type awareness hữu ích. citeturn12view4turn13view4

**Error budget dự kiến**, xếp theo expected recoverable impact:

| Error class | Evidence hiện có | Root cause khả dĩ | Candidate fix | Ưu tiên |
|---|---|---|---|---|
| Exact span boundary | 20,0 F1 gap EM↔IoU Arg-I | BIO/token decoding không tối ưu span | span-first/set extraction | **1** |
| Role confusion | 11,96 F1 Arg-I↔Arg-C | semantic overlap giữa roles | codebook prototypes + tuple context | **2** |
| Domain shift, nhất là DH | domain plot rất thấp | lexicon + discourse/rhetorical shift | hierarchical discourse + robust training | **3** |
| Missing arguments / recall | OneIE R=43,71; DEGREE chỉ 13,83 | candidate/generation recall | explicit span candidates | **4** |
| Rare Analysis/Ethical/Contradiction | 90/3/3 examples | extreme sample scarcity | definition shrinkage/prototypes | **5** |
| Conclusion event args | event-type plot yếu | implication/context ambiguity | type-conditioned queries | **6** |
| Trigger tuple errors | trigger F1 72,40 | composite structured trigger | explicit tuple component heads | **7** |

Không nên nói Context chắc chắn là role tạo nhiều false negatives nhất vì chúng ta chưa có baseline prediction dump để đếm FN. Tuy nhiên Context có **1.352 instances**, lớn nhất corpus, và role-wise F1 của nó không cao, nên đây là ứng viên có số FN tuyệt đối lớn nhất. Đây là inference, không phải measured fact. citeturn12view1turn12view4

### Trả lời các câu hỏi bắt buộc

| Câu hỏi | Kết luận hiện tại | Confidence / decisive experiment |
|---|---|---|
| Vì sao OneIE > DEGREE/EEQA Arg-C? | chủ yếu do recall; structured/joint modeling là hypothesis mạnh | Medium; cùng backbone + ablate decoder |
| Gain do global constraints/BERT/objective/preprocess? | chưa xác định causal | Low; factorial ablation |
| Boundary hay role khó hơn? | metric gap nghiêng về boundary: 20,0 vs 11,96 điểm | High |
| Upper bound với gold spans? | **chưa được paper báo** | chạy oracle role classifier |
| Gold event type upper bound? | chưa biết cho tuning model | type oracle on/off |
| Gold trigger upper bound? | chưa biết; metric argument không trực tiếp yêu cầu trigger match | trigger oracle conditioning |
| Role tạo nhiều FN tuyệt đối nhất? | Context là ứng viên mạnh nhưng chưa đo | Medium; count prediction dump |
| Domain tạo nhiều errors nhất? | Digital Humanities là ứng viên rõ từ F1 và có 120 docs | Medium-high |
| Domain gap do vocabulary hay structure? | chưa phân rã; nhiều khả năng cả hai | lexical masking + discourse probe |
| Longer context có giúp? | multi-sentence có lý, nhưng avg 2,95 câu không hỗ trợ “càng dài càng tốt” | Medium-low; 256/512/1024 controlled |
| Joint > pipeline? | paper gợi ý có, nhưng bị confound | Medium; same encoder/head experiment |
| Scientific encoder > stronger generic? | **chưa biết** | SciBERT vs modern generic matched head |
| Event type đủ hay cần domain? | event type rõ ràng hữu ích; explicit domain ID có nguy cơ shortcut | Medium |
| Position/discourse prior hữu ích? | có plausibility cao, nhưng nên soft | Medium |
| Rare role tăng mà không hại frequent roles? | có thể qua prototype/shrinkage | Low-medium trước experiment |
| Bỏ domain shortcut còn giữ performance? | chưa kiểm tra | leave-one-domain-out + domain marker masking |
| Metric nhạy boundary? | **có**, cả threshold và EM/IoU gap đều cho thấy | High |
| Thay đổi nhỏ nhất có thể tạo gain đáng tin? | span-set head trên encoder mạnh là candidate tốt nhất | Medium; ≥3 seeds |

## Bằng chứng từ literature và sàng lọc giả thuyết

SciEvent nằm ở giao điểm của event extraction, scientific IE, document-level modeling và low-resource structured prediction. Các công trình quan trọng nhất đối với thiết kế này không phải tất cả đều là “scientific event extraction”.

### Literature evidence matrix

| Công trình | Ý tưởng tạo gain | Giả định có phù hợp SciEvent? | Transfer được | Không nên copy máy móc |
|---|---|---|---|---|
| OneIE, ACL 2020 | joint graph IE + global features | có, vì event/role phụ thuộc lẫn nhau | structured scoring | ontology ACE constraints | citeturn19search11 |
| EEQA, EMNLP 2020 | EAE như QA | một phần | semantic role questions | pipeline trigger error propagation | citeturn19search0 |
| DEGREE, NAACL 2022 | conditional generation + templates | một phần | schema verbalization | autoregressive exact-span recall | citeturn19search4 |
| PAIE, ACL 2022 | prompted span selectors + bipartite matching | **rất phù hợp** | multi-argument span set | generic prompts không khai thác trigger tuple | citeturn19search3 |
| Contextualized Soft Prompts, Findings ACL 2023 | event-conditioned soft prompting | phù hợp | event-specific conditioning | prompt-only novelty | citeturn20search1 |
| SCPRG, Findings ACL 2023 | non-argument context + role relevance | phù hợp | role/context semantic relevance | không giải quyết structured tuple trực tiếp | citeturn20search2 |
| AMR doc-level EAE, ACL 2023 | graph/link structure | có thể | structural relations | AMR parser cost/domain error | citeturn20search11 |
| Global constraints + prompting, Findings EACL 2023 | label/global compatibility | một phần | soft compatibility | hard event-role bans không hợp SciEvent | citeturn20search9 |
| Document-level EE probing, Findings EMNLP 2023 | phân tích sentence-distance/context | rất phù hợp | distance diagnostics | không phải architecture hoàn chỉnh | citeturn20search6 |
| DEEIA, Findings ACL 2024 | dependency-guided/event-specific aggregation | có thể | event-specific context | dependency parser có thể domain-sensitive | citeturn20search12 |
| CMR EAE, COLING 2025 | compressive-memory retrieval | một phần | context retrieval | có thể quá phức tạp cho short abstracts | citeturn20search7 |
| SciERC/SciIE, EMNLP 2018 | span-based joint scientific IE | **có** | scientific span representation | schema không phải event tuple | citeturn21search4 |
| SciBERT, EMNLP 2019 | domain-specific scientific pretraining | có | backbone control | backbone swap không đủ novelty | citeturn21search0 |
| SciREX, ACL 2020 | document-level scientific IE | có | cross-sentence scientific structure | n-ary relation task khác | citeturn21search9 |
| SciER, EMNLP 2024 | scientific entity/relation IE + OOD analysis | có về domain robustness | OOD scientific diagnostics | relation objective không trực tiếp chuyển nguyên xi | citeturn21search1 |
| ITER, Findings EMNLP 2024 | efficient encoder-based structured extraction | có | efficient discriminative extraction | task structure khác | citeturn21search3 |

Hai kết luận từ literature nổi bật.

Thứ nhất, **PAIE-like span set extraction** phù hợp hơn BIO tagging hoặc free-form generation với metric SciEvent: nó trực tiếp dự đoán spans, hỗ trợ nhiều argument cùng role và dùng matching để tránh slot permutation. PAIE đã chứng minh giá trị của prompt-conditioned span selectors và bipartite matching, vì vậy “dùng span head” một mình **không phải novelty**. citeturn19search3

Thứ hai, novelty khả dĩ nằm ở cấu trúc riêng của SciEvent: trigger không phải một verb anchor mà là **Agent–Action–Object tuple**, trong khi roles là discourse-semantic roles và corpus cross-domain. Đây là chỗ chưa được khai thác đầy đủ bởi OneIE, DEGREE, EEQA hay PAIE.

### Các hypothesis đã xem xét

| Hypothesis | Expected gain | Novelty | ≤30 GB | Verdict |
|---|---:|---:|---:|---|
| H1 Hierarchical sentence/token discourse encoder | Medium | Medium | tốt | **giữ** |
| H2 Span-first set extraction | High | thấp nếu đứng một mình | tốt | **giữ như core engineering** |
| H3 Codebook role prototypes + contrastive shrinkage | Medium-high trên rare roles | Medium-high | rất tốt | **giữ** |
| H4 Domain-aware adapters/explicit domain ID | Medium | thấp-medium | tốt | **hạ ưu tiên** vì shortcut risk |
| H5 Tuple-anchored Agent–Action–Object conditioning | **High** | **High** | tốt | **ưu tiên cao nhất** |
| H6 Soft global/cardinality structured decoding | Medium | Medium | tốt | **giữ** |
| H7 Multi-task type + trigger tuple + arguments | Medium-high | Medium | tốt | **giữ** |
| H8 Domain-adaptive scientific pretraining | Medium | thấp | tốt | nhánh external-data riêng |
| H9 Train-only retrieval/prototypes | Medium | Medium | tốt | nhánh sau |
| H10 3B–7B QLoRA generation | bất định | thấp-medium | risky | **không chọn làm main line** |

### Các hướng bị loại khỏi contribution chính

**“Thay BERT-large bằng encoder mới.”** Đây là diagnostic bắt buộc, nhưng không đủ novelty cho ACL/EMNLP.

**“Tăng context lên 8k.”** Evidence SciEvent chỉ cho thấy event trung bình 2,95 câu; chưa có bằng chứng rằng context window là bottleneck chính. citeturn15view0

**“LoRA/QLoRA.”** Đây là resource technique, không phải đóng góp task-specific; hơn nữa phải báo riêng PEFT vs full fine-tuning.

**Hard event-role constraints.** Repo cho phép tất cả roles với cả bốn event types, nên constraint thủ công mạnh dễ inject false assumptions.

**Explicit domain-ID routing làm core method.** Có thể tăng in-domain score nhưng dễ học shortcut và làm yếu domain generalization. Domain label phù hợp hơn cho balanced/group-robust training objective chứ không nên là crutch ở inference.

**DAPT là novelty chính.** Scientific continued pretraining có thể tăng score nhưng tạo external-data regime và không giải quyết đặc trưng trigger/argument của SciEvent.

**Evidence:** span-set extraction và scientific pretraining đều đã có prior art rõ ràng. citeturn19search3turn21search0

**Interpretation:** contribution đủ mạnh phải giải quyết một dependency đặc thù của SciEvent thay vì chỉ ghép kỹ thuật phổ biến.

**Confidence:** medium-high.

**Falsifier:** nếu tuple-aware conditioning không vượt một span-set baseline matched-backbone qua nhiều seed, thành phần được cho là novel không có evidence thực nghiệm và paper phải hạ claim xuống engineering improvement.

## Phương pháp được chọn: TARS-SciEvent

Tên làm việc:

**TARS-SciEvent — Tuple-Anchored Rhetorical Span-set Extraction for Scientific Events**

Đây là candidate architecture được chọn, **chưa phải một hệ thống đã đạt 41,61+ trong experiment mới**.

### Research intuition

SciEvent có ba đặc điểm tương tác:

1. trigger là tuple Agent–Action–Object thay vì anchor đơn;
2. semantic arguments thường phụ thuộc ý nghĩa tổng thể của event và rhetoric của abstract;
3. role distribution rất lệch, tới mức hai roles chỉ có ba examples. citeturn12view1turn15view1

OneIE chủ yếu dùng structured joint scoring; PAIE cho thấy trực tiếp dự đoán span set hiệu quả cho EAE; scientific IE literature cho thấy span/document representations có ích. citeturn19search11turn19search3turn21search4turn21search9

TARS kết hợp chúng theo một dependency mới:

\[
\text{Trigger tuple}
\rightarrow
\text{event semantic query}
\rightarrow
\text{candidate argument spans}
\leftrightarrow
\text{role-definition prototypes}
\]

và thêm sentence-level rhetorical context nhưng không hard-code section/domain rules.

### Encoder và rhetorical representations

Cho event window gồm token \(x_1,\ldots,x_n\), encoder sinh

\[
H=(h_1,\ldots,h_n)=Enc_\theta(x_{1:n}).
\]

Với mỗi câu \(m\):

\[
s_m = Pool(\{h_i: sent(i)=m\})
\]

và một Transformer nhỏ trên các sentence representations:

\[
\tilde S=Trans_{\phi}(s_1,\ldots,s_M).
\]

Token representation được fusion:

\[
h_i' =
LN\left(h_i+
W_s\tilde s_{sent(i)}\right).
\]

Mục tiêu của layer này không phải “long context vì long context”, mà là cho model một representation explicit về câu nào đóng vai trò rhetorical context xung quanh event.

### Trigger tuple head

Bốn component heads dự đoán:

\[
c\in\{Agent,\ Action,\ PrimaryObject,\ SecondaryObject\}.
\]

Với predicted hoặc teacher-forced span \(C_c\):

\[
g_c=SpanPool(H',C_c).
\]

Secondary Object có learned missing vector nếu absent.

Tuple query:

\[
q_T =
LN\left(
W_T[
g_{Ag};
g_{Act};
g_{PO};
g_{SO};
e_{type}
]
\right).
\]

Có thể dùng gating:

\[
\alpha_c=\sigma(w_c^\top g_c+b_c),
\]

\[
q_T =
W_T[
\alpha_{Ag}g_{Ag};
\alpha_{Act}g_{Act};
\alpha_{PO}g_{PO};
\alpha_{SO}g_{SO};
e_{type}].
\]

Điều này cho model khả năng giảm tác động của một trigger component dự đoán không chắc chắn.

### Span-first candidate representation

Không dùng BIO làm representation chính. Với candidate span \(a=(i,j)\), \(j-i\le L_{\max}\):

\[
z_{ij}=
[
h_i';
h_j';
AttnPool(h_i',\ldots,h_j');
e_{width(j-i)};
e_{\Delta sent(i,T)}
].
\]

Boundary proposal head loại nhanh spans xác suất thấp trước role scoring để complexity không thành \(O(n^2R)\) toàn bộ.

Giữ top-\(K\) starts/ends hoặc top-\(K_s\) spans:

\[
\mathcal C = TopK_{i,j}\,
s_{boundary}(i,j).
\]

### Codebook-aware role prototypes

Mỗi role \(r\) có hai prototype:

- \(p_r^{def}\): embedding của định nghĩa chính thức trong annotation codebook;
- \(p_r^{train}\): centroid/learned prototype từ gold training arguments.

Kết hợp bằng frequency-aware shrinkage:

\[
p_r=
\lambda_r p_r^{def}
+(1-\lambda_r)p_r^{train},
\]

với ví dụ:

\[
\lambda_r=
\frac{\kappa}{\kappa+n_r}.
\]

Như vậy role rất hiếm sẽ dựa nhiều hơn vào semantic definition, trong khi frequent role được phép dựa nhiều hơn vào empirical representation.

Đây đặc biệt phù hợp với Ethical/Contradiction, nơi \(n_r=3\) toàn dataset. citeturn12view1

Span-role score:

\[
s(a,r)=
MLP(
[z_a;
q_T;
z_a\odot q_T;
\cos(W_pz_a,p_r);
e_{eventType};
e_{\Delta sentence}]
).
\]

Role-definition information đến từ **official codebook**, không phải test labels hoặc generated test demonstrations.

### Set decoder

SciEvent cho phép nhiều spans có thể cùng role; prompt annotation chính thức cũng không áp giả định “mỗi role tối đa một span”. citeturn13view4 Do đó decoder tạo \(K_r\) slots cho role \(r\), gồm NULL.

Cho gold set \(G_r=\{g_1,\dots,g_m\}\), training dùng permutation-invariant matching:

\[
\pi^\*=
\arg\min_{\pi}
\sum_{k=1}^{m}
C(\hat a_{\pi(k)},g_k).
\]

Set loss:

\[
\mathcal L_{set}
=
\sum_r
\left[
\sum_{k=1}^{m_r}
-\log p(g_k,r\mid q_T)
+
\sum_{k>m_r}
-\log p(NULL)
\right].
\]

Đây kế thừa tinh thần bipartite span assignment của PAIE, nhưng conditioning query và role representation khác về bản chất. citeturn19search3

### Auxiliary objectives

Prototype contrastive loss:

\[
\mathcal L_{proto}
=
-\sum_a
\log
\frac{
\exp(sim(Wz_a,p_{r_a})/\tau)
}{
\sum_{r'}
\exp(sim(Wz_a,p_{r'})/\tau)
}.
\]

Trigger component loss:

\[
\mathcal L_{trg}
=
\sum_c
(
CE(start_c)+CE(end_c)
).
\]

Event type:

\[
\mathcal L_{type}=CE(\hat y,y).
\]

Boundary proposal:

\[
\mathcal L_{bnd}
=
CE(\hat b^{start},b^{start})
+
CE(\hat b^{end},b^{end}).
\]

Final core objective:

\[
\boxed{
\mathcal L=
\mathcal L_{set}
+
\lambda_b\mathcal L_{bnd}
+
\lambda_t\mathcal L_{trg}
+
\lambda_e\mathcal L_{type}
+
\lambda_p\mathcal L_{proto}
}
\]

Domain robustness là một **optional ablation**, không mặc định vào core novelty. Có thể dùng domain-balanced batches hoặc worst-group weighting mà không feed domain ID cho inference.

### Kiến trúc tổng thể

```text
              scientific event window
                       │
                       ▼
             pretrained encoder
                       │
             ┌─────────┴─────────┐
             │                   │
      token representations   sentence pooling
             │                   │
             │            rhetorical transformer
             └─────────┬─────────┘
                       ▼
                fused token states
             ┌─────────┴──────────┐
             │                    │
       trigger tuple heads   boundary proposals
 Agent / Action / PO / SO          │
             │                     ▼
             ▼               candidate spans
       tuple semantic query        │
             │              ┌──────┴──────┐
             │              │             │
             └──────────► span-query   role prototype
                            scoring       bank
                              │             │
                              └──────┬──────┘
                                     ▼
                             set/Hungarian decoder
                                     │
                                     ▼
                    Context / Purpose / Method / ...
```

### Novelty claim chart

| Component | Closest work | Prior art | TARS change | Novelty status |
|---|---|---|---|---|
| span-set decoder | PAIE | prompt span selectors + matching | tuple-conditioned scientific spans | component không mới |
| joint/global structure | OneIE | global graph features | tuple → span dependency explicit | khác formulation |
| contextual role semantics | SCPRG | role relevance/context clues | official-codebook prototype bank | incremental alone |
| scientific representations | SciBERT/SciIE | domain encoder/span IE | only backbone/control | không mới |
| sentence rhetoric | doc-level EAE/SciREX | document context | lightweight event-window hierarchy | moderate |
| **tuple-anchored argument query** | trigger-conditioned EAE broadly | thường anchor/event embedding | **compositional Agent–Action–PO–SO query** | **core novelty** |
| **frequency-adaptive definition prototype** | prototype/semantic label methods broadly | label semantics | rare SciEvent roles shrink toward annotation definitions | secondary novelty |
| **tuple + prototype + permutation-invariant span set** | chưa thấy trong located SciEvent work | các thành phần tồn tại riêng | dependency được thiết kế cho SciEvent schema | **paper claim cần ablation bảo vệ** |

Contribution không nên viết là “we introduce span extraction” hay “we use contrastive learning”. Claim hợp lý hơn là:

> **TARS models scientific-event arguments as a set of spans jointly anchored to SciEvent’s compositional Agent–Action–Object trigger, while regularizing extremely sparse scientific discourse roles toward their annotation-semantic prototypes.**

Đây là claim có thể sống sót reviewer hơn, nhưng chỉ nếu ablation chứng minh tuple conditioning và prototype regularization đều có effect vượt matched-backbone span baseline.

### VRAM budget

Vì không có CUDA GPU tương ứng trong execution environment, những số sau là **memory accounting estimates**, không phải measured VRAM.

Giả sử backbone khoảng **395M parameters**, full fine-tuning bf16 và conventional Adam:

| Persistent component | Approx. memory |
|---|---:|
| bf16 parameters | ~0,79 GB |
| bf16 gradients | ~0,79 GB |
| Adam fp32 m + v | ~3,16 GB |
| optional fp32 master weights | ~1,58 GB |
| **persistent subtotal** | **~6,32 GB** |
| activations / attention / CUDA workspace | phụ thuộc batch & length |
| TARS heads | nhỏ so với backbone |

Với backbone khoảng 304M, persistent subtotal tương tự vào khoảng **4,86 GB**. Đây không chứng minh model fit: activation memory ở backward mới là biến số cần đo.

Primary configuration nên bắt đầu:

```yaml
precision: bf16
micro_batch_size: 1
gradient_accumulation: 16
max_length: 512
gradient_checkpointing: true
attention: sdpa_or_flash_if_supported
optimizer: adamw
full_finetuning: true
target_peak_vram_gib: 28
```

Sau đó tăng micro-batch/length chỉ nếu:

```python
torch.cuda.reset_peak_memory_stats()
...
peak_alloc = torch.cuda.max_memory_allocated() / 2**30
peak_reserved = torch.cuda.max_memory_reserved() / 2**30

assert peak_alloc <= 30.0
```

và đồng thời ghi `nvidia-smi --query-compute-apps=...`.

Tôi khuyến nghị safety gate thực tế là:

\[
PeakAllocated \le 28\ \text{GiB}
\]

chứ không thiết kế sát 29,99 GB.

Nếu AdamW đầy đủ vượt budget, thứ tự giảm memory nên là:

1. gradient checkpointing;
2. micro-batch 1 + accumulation;
3. length bucketing;
4. efficient attention;
5. optimizer 8-bit.

Dùng optimizer 8-bit **không biến experiment thành PEFT** nếu mọi model weights vẫn trainable; nhưng phải disclosure vì optimizer regime khác.

QLoRA 3B–7B nên được ghi vào bảng khác hoàn toàn:

```text
Regime = PEFT / QLoRA
≠ full fine-tuning
```

### Implementation plan

Không nên sửa baseline code đến mức mất khả năng exact reproduction. Giữ original paths read-only và bổ sung:

```text
src/
  data/
    scievent.py
    spans.py
  tars/
    encoder.py
    discourse.py
    tuple_query.py
    prototypes.py
    span_set.py
    decoder.py
    model.py
  eval/
    official_wrapper.py
    strict_evaluator.py

scripts/
  audit_split.py
  audit_duplicates.py
  audit_dataset_stats.py
  build_strict_split.py
  run_oracles.py
  measure_vram.py
  train_tars.py
  predict_tars.py

tests/
  test_official_metric.py
  test_span_conversion.py
  test_split_disjoint.py
  test_trigger_tuple.py
  test_duplicate_predictions.py

configs/
  scievent/
    oneie_reproduction.yaml
    span_baseline.yaml
    tars_core.yaml
    tars_full.yaml
```

Không thay `data_scripts/shared/split_data.py`. Thay vào đó thêm `build_strict_split.py`, để mọi người vẫn có thể reproduce exact historical behavior.

`official_wrapper.py` phải gọi evaluator chính thức cho headline numbers nhưng trước đó assert prediction-ID set, và chạy thêm corrected evaluator. Nếu hai score khác nhau, report:

```text
OFFICIAL score
STRICT/CORRECTED score
```

chứ không âm thầm thay metric.

**Expected failure cases:** trigger tuple prediction sai gây conditioning noise; implicit arguments không có clean span; Ethical/Contradiction quá ít để học kể cả prototype; discourse prior overfit domain; candidate pruning bỏ mất long arguments; cùng text có nhiều plausible role; set slots không đủ cho repeated arguments.

**Falsification criterion:** nếu `TARS-core – matched span baseline` có 95% paired bootstrap CI chứa 0 một cách rộng qua ≥3 seeds, claim rằng tuple anchoring giải quyết dependency đặc thù SciEvent phải bị bác bỏ.

## Experimental protocol, reproduction package và SOTA gate

### Baseline reproduction status

Điểm này cần cực kỳ rõ ràng:

| Hạng mục | Published | Reproduced trong audit này |
|---|---:|---:|
| OneIE Trigger F1 | 72,40 | **chưa chạy GPU** |
| OneIE Arg-I IoU | 53,57 | **chưa chạy GPU** |
| OneIE Arg-C IoU | 41,61 | **chưa chạy GPU** |
| OneIE Arg-C EM | 26,61 | **chưa chạy GPU** |
| Exact evaluator behavior | paper/code | **đã reverse-engineer + synthetic tests** |
| Exact official split identity | paper nói document 80/10/10 | **chưa xác nhận; code discrepancy** |
| Peak VRAM | không phải headline paper result | **chưa đo** |

Published numbers đến trực tiếp từ SciEvent paper. citeturn12view2turn12view3turn15view2

Vì vậy đây **không phải “baseline reproduced successfully”**. Theo chính rule của user, model development có thể được thiết kế, nhưng SOTA evaluation không được coi là hợp lệ cho tới khi reproduction vượt gate.

### Thứ tự experiment tối ưu chi phí

**Reproduction gate**

```text
R0  Exact OneIE config, official-code split, seed behavior matched
R1  Same under strict document split
R2  Verify official evaluator byte-for-byte
R3  Record CUDA peak + runtime
```

Nếu R0 lệch paper >1 absolute F1 point, dừng model comparison và kiểm tra split, CSCW, tokenizer/checkpoint revisions, package versions và evaluator.

**Oracle diagnostics**

Đây là nhóm experiment có information gain cao nhất:

```text
O1  predicted span + predicted role                normal system
O2  GOLD argument spans -> predict role            role upper bound
O3  GOLD event type                                quantify type error
O4  GOLD trigger tuple                             quantify trigger-conditioning error
O5  GOLD type + GOLD trigger                       extraction ceiling conditional on spans
```

O2 trả lời trực tiếp “bao nhiêu lỗi là boundary vs classification”, tốt hơn suy luận từ EM/IoU gap.

**Cheap architecture diagnostics**

```text
D1  same span head + SciBERT
D2  same span head + strong generic encoder A
D3  same span head + strong generic encoder B

D4  BIO head
D5  independent span head
D6  set/Hungarian span head

D7  no event-type conditioning
D8  gold/predicted event-type conditioning
```

Đây là nơi chọn backbone; không dùng test set.

**Hypothesis experiments**

| ID | Model | Câu hỏi |
|---|---|---|
| H0 | matched span-set baseline | reference mới |
| H1 | + tuple query | structured trigger có thực sự giúp? |
| H2 | + definition prototype | role semantics có giúp? |
| H3 | + frequency shrinkage | rare-role gain có ổn định? |
| H4 | + rhetorical hierarchy | sentence structure có giúp? |
| H5 | + balanced/group-robust training | domain generalization có tốt hơn? |
| FULL | H1+H2/H3+chỉ các component thắng | final |

Chỉ combine component nếu dev improvement được lặp lại qua ≥2 seeds ở funnel stage.

### Ablation table phải được điền theo cấu trúc này

Không có số mới nên không được giả lập một bảng “đẹp”.

| Variant | Arg-I EM | Arg-I IoU | Arg-C EM | Arg-C IoU | DH F1 | Rare-role F1 | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched encoder + BIO | TBD | TBD | TBD | TBD | TBD | TBD | measured |
| span-set only | TBD | TBD | TBD | TBD | TBD | TBD | measured |
| + tuple anchoring | TBD | TBD | TBD | TBD | TBD | TBD | measured |
| + role definitions | TBD | TBD | TBD | TBD | TBD | TBD | measured |
| + frequency shrinkage | TBD | TBD | TBD | TBD | TBD | TBD | measured |
| + discourse hierarchy | TBD | TBD | TBD | TBD | TBD | TBD | measured |
| + robustness objective | TBD | TBD | TBD | TBD | TBD | TBD | measured |
| **TARS full** | TBD | TBD | TBD | TBD | TBD | TBD | measured |

“TBD” ở đây là khoa học hơn bất kỳ fabricated SOTA result nào.

### Statistical validation

Final configuration phải được freeze trên train/dev trước khi test.

Tối thiểu:

```text
seeds = [13, 42, 101]
```

tốt hơn:

```text
seeds = [13, 21, 42, 87, 101]
```

Báo cáo:

\[
mean,\quad std,\quad median,\quad best
\]

nhưng headline comparison phải dựa trên mean/CI, không chọn “best test seed”.

Vì nhiều event windows có thể thuộc cùng document, paired bootstrap nên **resample theo document**, không resample độc lập từng event. Điều đó bảo toàn within-document dependence.

Một final gain nên có:

- positive mean Arg-C gain;
- 95% document-block bootstrap CI hợp lý;
- không chỉ tăng ở NLP;
- không chỉ tăng frequent Method/Result roles;
- không phá rare roles;
- không dựa trên metric artifact;
- measured VRAM ≤30 GB cho mọi final run.

### Các experiment để phân rã domain shift

Để trả lời vocabulary-shift vs discourse-shift thay vì phỏng đoán:

**Lexical probe:** huấn luyện classifier dự đoán domain từ token/span embeddings; đo domain separability.

**Masking probe:** mask venue/journal/domain markers và specialized terms; nếu gap giảm mạnh, lexical shortcuts quan trọng.

**Structure-only probe:** thay lexical span representations bằng sentence position, length, relative trigger distance và rhetorical features; nếu vẫn dự đoán domain/event role tốt, structural shift tồn tại.

**Leave-one-domain-out:** train bốn domain, dev/test domain thứ năm. Đây không phải official leaderboard result mà là robustness analysis.

**Matched-terminology split:** stratify examples theo scientific term frequency để xem unseen terminology ảnh hưởng ra sao.

### Reproducibility registry

Mỗi run phải ghi ít nhất:

```text
experiment_id
git_commit
data_protocol = official_code | strict_document
split_sha256
train_doc_hash
dev_doc_hash
test_doc_hash
evaluator_commit
backbone_checkpoint_revision
full_ft | peft
external_data_manifest
seed
precision
max_length
micro_batch
gradient_accumulation
optimizer
lr
epochs
early_stopping_rule
best_dev_checkpoint
thresholds
train_metric
dev_metric
test_metric
peak_memory_allocated
peak_memory_reserved
nvidia_smi_peak
runtime
checkpoint_sha256
notes
```

Các hash của IDs đặc biệt quan trọng vì chỉ nói “80/10/10 split” là không đủ sau phát hiện trong `split_data.py`.

### Command package đề xuất

Exact environment versions phải được pin theo reproduction run chứ không nên bịa ra từ môi trường chưa chạy. Skeleton:

```bash
# Audit before any training
python scripts/audit_split.py \
  --train data/train.json \
  --dev data/dev.json \
  --test data/test.json \
  --check-doc-id \
  --check-wnd-id \
  --check-text-duplicates \
  --fail-on-overlap

python scripts/audit_dataset_stats.py \
  --data SciEvent_data/annotated/event_extraction_finetune_model.jsonl \
  --output artifacts/dataset_stats.json

# Official OneIE reproduction
python baselines/ONEIE/train.py \
  --config baselines/ONEIE/config/config_oneie_scievent.json

# TARS development: TEST inaccessible during selection
python scripts/train_tars.py \
  --config configs/scievent/tars_core.yaml \
  --seed 42

# Final frozen inference
python scripts/predict_tars.py \
  --config artifacts/frozen_final_config.yaml \
  --checkpoint checkpoints/tars_seed42.pt \
  --split test \
  --output predictions/tars_seed42.json

# Official evaluator wrapper
python src/eval/official_wrapper.py \
  --gold data/test.json \
  --pred predictions/tars_seed42.json \
  --validate-ids \
  --report-official \
  --report-strict

# Actual hardware gate
python scripts/measure_vram.py \
  --run-manifest artifacts/final_run.json
```

### SOTA gate sau audit

| Điều kiện | Trạng thái 22/09/2026 |
|---|---|
| latest SciEvent literature searched | ✅ đã thực hiện; không tìm thấy comparable successor |
| identical protocol established | ⚠️ **split discrepancy chưa giải quyết** |
| official evaluator audited | ✅ cho extraction contract |
| test set không dùng để tune | ✅ trong thiết kế mới; chưa có training run |
| vượt comparable best result | ❌ chưa chạy |
| multiple seeds | ❌ |
| train/dev/test leakage ruled out | ❌ cần empirical ID intersections |
| measured VRAM ≤30 GB | ❌ chưa có CUDA measurement |
| reproducible code/config/checkpoint | ❌ candidate design mới |
| external data disclosed | ✅ policy đã định nghĩa, chưa có new run |

Do đó câu khoa học chính xác hiện tại là:

> **TARS-SciEvent là một candidate SOTA architecture được thiết kế từ audit benchmark và error structure, chưa phải SOTA đã được chứng minh. Mốc tuning-based công bố mạnh nhất mà cuộc điều tra này xác minh được vẫn là OneIE với 41,61 Arg-C IoU F1, nhưng khả năng so sánh của con số này phải được xác nhận lại sau khi giải quyết bất nhất giữa document-level split được paper mô tả và event-type/window-level split trong code.** citeturn12view3turn15view0

Open scientific issues quan trọng nhất còn lại không phải “thử thêm architecture”, mà theo thứ tự là: **xác minh exact published split**, **tái tạo OneIE trong ±1 point**, **đo oracle boundary/role ceilings**, rồi mới quyết định liệu tuple-conditioned span sets có thật sự là hướng thắng.

Paper-ready contribution statement, nếu và chỉ nếu experiments cuối cùng ủng hộ hypothesis, nên được viết ở mức:

> *We introduce TARS-SciEvent, a structured scientific event extractor that represents SciEvent triggers compositionally as Agent–Action–Object tuples and uses the resulting event representation to anchor permutation-invariant argument span-set decoding. To address the benchmark’s extreme discourse-role sparsity, TARS regularizes learned role representations toward annotation-codebook semantics using frequency-adaptive prototypes. Unlike hard domain- or event-role constraints, the model retains soft cross-domain representations and is evaluated under both the historical SciEvent protocol and a verified document-disjoint protocol.*

Nếu tuple anchoring hoặc prototype ablation không tạo gain ổn định, các mệnh đề tương ứng phải bị xóa khỏi contribution claim thay vì được bảo vệ hậu nghiệm.