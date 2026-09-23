# Experiment record — CARVE

Exact method specification and every measured number.

Purpose: a single reference containing **every number we measured**, with the
exact configuration that produced it, written to be transcribed into a paper
without re-deriving anything.

Two conventions used throughout:

- Numbers are quoted with the **decoding rule and tuning grid** that produced
  them, because the same checkpoint scores differently under different grids.
- Where a comparison is **confounded**, it is marked ⚠ and the confound is
  stated. Do not put ⚠ rows into a paper as clean ablations. See §9.

Hardware for every run: 1× NVIDIA RTX 5880 Ada Generation (49 GB), CUDA 12.8,
PyTorch 2.11.0+cu128, transformers 5.5.4, Python 3.13.5.

---

## 1. Task, data, and the evaluation contract

### 1.1 Data

Read-only from `third_party/SciEvent/SciEvent_data/ONEIE/all_splits/`:

| split | windows | documents |
|---|---|---|
| train | 1278 | 493 |
| dev | 158 | 133 |
| test | 163 | 147 |
| **total** | **1599** | **500** |

A *window* is one annotated segment of a scientific abstract. Each window carries
exactly one event mention.

**Domain distribution (windows):**

| split | ACL | bioinfo | cscw | dh | jmir |
|---|---|---|---|---|---|
| train | 244 | 224 | 292 | 261 | 257 |
| dev | 38 | 33 | 27 | 36 | 24 |
| test | 27 | 28 | 44 | 30 | 34 |

**Event-type distribution (windows):**

| split | Background/Introduction | Methods/Approach | Results/Findings | Conclusions/Implications |
|---|---|---|---|---|
| train | 390 | 387 | 305 | 196 |
| dev | 48 | 48 | 38 | 24 |
| test | 50 | 49 | 39 | 25 |

**Window length (whitespace tokens):** train mean 62.0, p95 123, max 260;
dev mean 62.7, p95 128, max 175; test mean 65.2, p95 143, max 195.

**Sub-token length (DeBERTa-v3 SentencePiece, `is_split_into_words=True`):**
train max 523 (exactly 1 window exceeds 512), dev max 225 / p99 216,
test max 240 / p99 235. Sequence length is not a constraint.

### 1.2 Argument roles and their support

The annotation has 12 role labels. The official evaluator **excludes**
`{Agent, PrimaryObject, SecondaryObject}` from Arg-I / Arg-C; those three are
used only to build the trigger ROUGE-L tuple. Training-split support:

| role | train instances | scored by Arg-I/Arg-C? |
|---|---|---|
| Agent | 1276 | no (ROUGE-L tuple only) |
| PrimaryObject | 1214 | no (ROUGE-L tuple only) |
| SecondaryObject | 104 | no (ROUGE-L tuple only) |
| Context | 1053 | **yes** |
| Method | 968 | **yes** |
| Results | 911 | **yes** |
| Challenge | 406 | **yes** |
| Purpose | 254 | **yes** |
| Implications | 235 | **yes** |
| Analysis | 67 | **yes** |
| Contradictions | 2 | **yes** |
| Ethical | 1 | **yes** |

Note the label strings in the released data are `Results`, `Implications`,
`Contradictions` (plural), whereas the paper's prose uses *Result*,
*Implication*, *Contradiction*. Use the data spellings in code.

### 1.3 Measured data geometry (train split, 1278 windows)

This is the measurement the whole method follows from.

| property | value |
|---|---|
| events per window | **exactly 1**, in all 1278 (and all 1599 across splits) |
| scored arguments per window | mean **3.05**, range 0–24; mode 2 (287 windows), then 1 (279), 3 (222), 4 (138), 0 (102) |
| windows with overlap **within** scored roles | **31 / 1278 = 2.4 %** |
| windows with overlap **within** Agent/Primary/Secondary | **2 / 1278 = 0.2 %** |
| windows with **cross-group** overlap (AAO vs scored) | **75 / 1278 = 5.9 %** |
| trigger overlaps a scored span | 0.6 % |
| token coverage of a window by scored spans | **0.563** |

**Gap between consecutive scored spans (token count → frequency):**
0 → **942**, 1 → 658, 2 → 271, 3 → 160, 4 → 105, 5 → 90, 6 → 63, 7 → 46,
8 → 41, 9 → 43. The modal gap is **zero**.

**Span length in tokens (train):**

| role | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| Agent | 1276 | 2.4 | 2 | 1 | 5 |
| PrimaryObject | 1214 | 5.9 | 4 | 2 | 13 |
| SecondaryObject | 104 | 6.4 | 5 | 2 | 15 |
| Context | 1053 | 10.2 | 8 | 3 | 21 |
| Method | 968 | 13.2 | 11 | 3 | 26 |
| Results | 911 | 14.5 | 13 | 6 | 25 |
| Challenge | 406 | 14.4 | 13 | 5 | 25 |
| Purpose | 254 | 12.5 | 11 | 6 | 22 |
| Implications | 235 | 12.9 | 11 | 4 | 23 |
| Analysis | 67 | 15.5 | 15 | 8 | 23 |
| Contradictions | 2 | 11.5 | 12 | 7 | 16 |
| Ethical | 1 | 5.0 | 5 | 5 | 5 |

**Trigger span length (train):** 1 token → 909, 2 → 154, 3 → 144, 4 → 49,
5 → 15, 6 → 5, 7 → 1, 8 → 1.

**Windows where a scored role occurs more than once:** Context 252, Method 219,
Results 211, Challenge 89, Implications 46, Purpose 42, Analysis 11.

**Interpretation for the paper.** Scored arguments are clause-sized
(10–15 tokens), near-contiguous (modal gap 0), and essentially non-overlapping
(97.6 %). This is *span segmentation with semantic-role labelling*, not
ACE-style entity-mention extraction. Roles repeat within a window, which breaks
one-answer-per-question QA formulations.

### 1.4 The official evaluation contract

Taken from reading `third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py`,
not from the paper's prose:

1. Arg-I and Arg-C are **trigger-insensitive** but **event-type sensitive**
   (`if p[0][2] != g[0][2]: continue`).
2. `{Agent, PrimaryObject, SecondaryObject}` are **excluded** from Arg-I/Arg-C.
3. Matching is **greedy one-to-one** per window; a gold span is consumed once.
4. Four matching modes: Exact, simple overlap (`inter > 0`), SciREX
   (`inter / pred_len > 0.5`), IoU (`inter / union > 0.5`). Headline is IoU.
5. Trigger "ROUGE-L" is **not** trigger-span overlap. It is ROUGE-L (with
   stemming) between the gold and predicted *concatenated strings*
   `"<Agent> <trigger> <PrimaryObject> <SecondaryObject>"`, macro-averaged over
   windows where at least one of the two strings is non-empty. Only the **first**
   occurrence of each of Agent/Primary/Secondary is used, and the trigger is
   read from `roles[0][0]`.

**Consequence used by the method.** Because each window holds exactly one event
and matching is event-type sensitive, a misclassified window type annihilates
*every* argument in it — simultaneously as false positives and false negatives.
Window type acts as a multiplier on the whole score.

### 1.5 Verification of the evaluation path

We import the upstream scorer and call **its own functions**; we never
reimplement a metric. Two checks:

- **Exporter oracle.** Gold spans routed through our own prediction writer into
  the official evaluator score **100.00** on Arg-C IoU, Arg-I IoU, Arg-C EM and
  ROUGE-L, on **both dev and test**.
- **CLI cross-check.** Running the upstream script exactly as the benchmark
  README prescribes on our frozen test predictions (seed 13) reproduces our
  reported numbers to the last decimal — see §7.1.

---

## 2. Method (exact specification)

Name: **CARVE** (*Clause-level Argument Recovery Via sEgmentation*). Architecture tag: multi-task word-level
span tagger with event-type conditioning.

### 2.1 Backbone

| item | value |
|---|---|
| checkpoint | `microsoft/deberta-v3-large` |
| HF revision | `64a8c8eab3e352a784c658aef62be1662607476f` |
| layers / hidden / heads / vocab | 24 / 1024 / 16 / 128 100 |
| total parameters in our model | **434.6 M** (≈304 M encoder blocks + ≈131 M embeddings + ≈1.5 M heads) |
| licence | MIT |
| adaptation | **fully fine-tuned**; no freezing, no LoRA/PEFT/adapters |
| precision | **fp32 master weights**, bf16 `torch.autocast` compute |
| tokenizer | SentencePiece, `is_split_into_words=True` over the benchmark's whitespace tokens |
| token→word pooling | **first sub-token** of each word |

No external corpora, pseudo-labels, synthetic data, LLM supervision or
retrieval were used. Training data is exactly the 1278 official training windows.

### 2.2 Heads

Let `H = 1024`, and let `w ∈ R^{B×W×H}` be the word-level states after
first-sub-token gathering (padded words masked to zero).

**(a) Attention pooling** → window vector
```
s      = Linear(H → 1)(w).squeeze(-1)              # B×W
s      = s.masked_fill(¬word_mask, finfo(dtype).min)
α      = softmax(s, dim=-1)
pooled = Σ_i α_i · w_i                              # B×H
```

**(b) TYPE head** — 4-way window event-type classification
```
type_logits = Linear(H → 4)(pooled)
```

**(c) Event-type conditioning** — a learned embedding added back to word states
```
type_emb : Embedding(4, H), initialised to ZEROS
cond     = gold_type   with probability p_teacher
         = argmax(type_logits)  otherwise          # scheduled sampling
w_cond   = w + type_emb(cond).unsqueeze(1)
```
At inference `cond = argmax(type_logits)` always. Zero-initialising `type_emb`
means the conditioning starts as an identity and is learned only if useful.

**(d) ROLE head** — word-level BIO over the **9 scored roles** (19 labels:
`O` + `B-`/`I-` × 9), applied to the **conditioned** states
```
role_logits = Linear(H → 19) ∘ Dropout ∘ GELU ∘ Linear(H → H)  (w_cond)
```

**(e) AAO head** — word-level BIO over `{Agent, Action, PrimaryObject,
SecondaryObject}` (9 labels), applied to the **unconditioned** states
```
aao_logits = Linear(H → 9) ∘ Dropout ∘ GELU ∘ Linear(H → H)  (w)
```
`Action` is the gold trigger span. This head exists **only** to construct the
ROUGE-L tuple; the Arg-I/Arg-C metrics ignore every label it produces.

**Why two heads and not one.** Measured overlap: within scored roles 2.4 %,
within Agent/Primary/Secondary 0.2 %, but **across the two groups 5.9 %**. A
single BIO layer would have to delete labels on 5.9 % of windows; two disjoint
layers do not.

### 2.3 BIO construction (training targets)

`spans_to_bio` — deterministic, and the tie-break rule matters:

1. Sort spans by `(length, start)` **ascending** — shorter spans win conflicts,
   because the shorter annotation is the more specific one and a truncated long
   span often still clears IoU > 0.5.
2. For each span, take only the **first contiguous run** of not-yet-claimed
   tokens inside it; label that run `B-`/`I-`; mark those tokens claimed.
3. Words outside the truncated sequence window get label `-100` (ignored).

**Measured cost of this encoding:** BIO round-trip recovers **3857 / 3897 =
98.97 %** of gold scored role spans exactly.

**Measured ceiling of the whole formulation (dev):** gold → BIO → decode →
official evaluator gives **Arg-C IoU 99.70**, Arg-I IoU 99.70, Arg-C EM 99.70,
ROUGE-L 100.00. The formulation costs ≈0.3 F1; it is not the bottleneck.

### 2.4 Decoding (`bio_to_spans`)

Standard BIO decode, tolerant of `I-` without `B-`: a new span starts on `B-` or
on any type change. A `B-` immediately after an `I-` of the **same** type starts
a new span, which is what allows two adjacent same-role gold spans (modal gap 0)
to be represented at all.

### 2.5 Loss

```
L = w_role · CE(role_logits, role_targets)
  + w_aao  · CE(aao_logits,  aao_targets)
  + w_type · CE(type_logits, type_targets)
```
Cross-entropy with `ignore_index = -100`, `label_smoothing = 0.0`.
Final weights: **w_role = 1.0, w_aao = 0.5, w_type = 0.5**.

### 2.6 Scheduled sampling schedule

```
p_teacher(epoch) = max( type_teacher_min , 1 − (epoch − 1)/(epochs − 1) )
type_teacher_min = 0.5
```
Epoch 1 conditions entirely on gold type; by the final epochs it conditions on
the model's own prediction half the time, matching inference.

### 2.7 Optimisation

| hyper-parameter | value |
|---|---|
| optimiser | AdamW, `weight_decay = 0.01` |
| lr (encoder) | **1e-5** |
| lr (heads) | **1e-4** |
| layer-wise lr decay | **1.0 (disabled)** — tested and rejected, §5.2 |
| schedule | linear warmup 10 % of total steps, then linear decay to 0 |
| epochs | **30** |
| batch size | **8** (train), 16 (eval); no gradient accumulation |
| max sequence length | 640 sub-tokens |
| dropout | 0.1 |
| gradient clipping | global norm 1.0 |
| seeds | **42, 13, 101** |
| runtime | ≈18 s/epoch, ≈9 min per full run |

### 2.8 Metric-aware decoding (three parameters, tuned on dev only)

For each decoded span, define its **confidence** as the mean posterior of the
assigned label over the span's tokens:
```
conf(s,e,t) = (1/(e−s)) · Σ_{i=s}^{e−1} P(label_i = argmax_i)
```
Then apply, in order:

1. **threshold** — drop spans with `conf < τ`;
2. **min length** — drop spans with `e − s < min_len`;
3. **merge** — join two same-role spans separated by `≤ merge_gap` tokens
   (confidence recombined as a length-weighted mean).

**FROZEN RULE: τ = 0.90, min_len = 3, merge_gap = 0.**

`merge_gap = 0` means merging is **off** — selected by the grid, which is itself
a finding: fragmentation was never the dominant error (see §5.1).

The AAO head is **not** thresholded, so trigger ROUGE-L is identical under
calibrated and uncalibrated decoding.

### 2.9 Checkpoint selection

Per epoch, dev predictions are decoded at every τ in
`{0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95}` with `min_len = 3`, and the
epoch's score is the **maximum dev Arg-C IoU F1** over that sweep. The best
epoch by that criterion is kept. Selection therefore uses the same family of
decoding rule that the final system ships, rather than argmax decoding.

### 2.10 How the frozen decoding rule was chosen (seed-agnostic)

`scripts/freeze_rules.py`: for each `(τ, min_len)` on the grid
`τ ∈ {0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95}`,
`min_len ∈ {2, 3, 4}`, compute **single-model** dev Arg-C IoU F1 for each of the
three seeds and take the **mean across seeds**. Pick the argmax.

This avoids fitting the threshold to whichever seed happens to look best on dev.
Selected: **τ = 0.90, min_len = 3**, dev Arg-C IoU **47.14 ± 0.26**.

### 2.11 Explicit modelling assumptions

- **One event per window.** Holds on all 1278 training windows, so it is
  derivable from training data alone. The model always emits exactly one event.
- **Gold segmentation is consumed**, as OneIE / DEGREE / EEQA do. Segmentation
  is not modelled and no segmentation number is reported.
- **The segment ordinal is deliberately unused** (see §8.3).
- **Event type is predicted, never given** (91.41 % accurate on test).

### 2.12 Implementation note that will bite any reimplementation

`transformers ≥ 5` honours the dtype stored in a checkpoint, and
`microsoft/deberta-v3-large` ships **fp16** weights. Training then updates fp16
master weights and AdamW produces NaN on the **first** optimizer step. Observed
signature: gradients finite, global grad-norm 11.02, then **all 390 parameter
tensors NaN after one `opt.step()` at lr 6.25e-07**. Fix: load the encoder with
`dtype=torch.float32` and take mixed precision from autocast only.

---

## 3. Experiment funnel — chronological record

| stage | what | outcome |
|---|---|---|
| 0 | read official evaluator; measure data geometry | reframed the task (§1.3) |
| 1 | correctness tests | oracle 100.00; BIO ceiling 99.70 |
| — | infrastructure defect | fp16-checkpoint NaN, fixed (§2.12) |
| 2 | smoke run, 5 epochs | dev Arg-C IoU 33.20, still rising → promote |
| 3 | H1 full run, 40 epochs | dev Arg-C IoU 38.88 argmax |
| 4 | diagnostics | 784 predicted spans vs 497 gold → calibration problem |
| 5 | H5 decoding calibration | 38.88 → 48.00 dev |
| 6 | H2/H4/H8 ablations | all rejected (§5) |
| 7 | 3 final seeds | dev 47.14 ± 0.26 under the frozen rule |
| 8 | H7 ensembling | rejected (§5.4) |
| 9 | freeze rule, evaluate test **once** | §6 |
| 10 | leakage/protocol audit | §7 |
| 11 | document-disjoint robustness probe | §7.4 |

### 3.1 Run registry

Dev Arg-C IoU below is each run's **own selection-time** score (per-epoch τ
sweep), which is *not* comparable across runs; the comparable re-scored numbers
are in §5.

| run | backbone | seed | epochs | lr enc/head | llrd | crf | span-head | best ep | dev Arg-C | dev Arg-I | dev Arg-C EM | dev ROUGE-L | type acc | runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `h1_deberta_s42` | deberta-v3-large | 42 | 40 | 1e-5/1e-4 | 1.0 | — | — | 20 | 38.88¹ | 47.46 | 24.51 | 79.06 | 89.2 | 763 s |
| `h3_llrd_s42` | deberta-v3-large | 42 | 30 | 2e-5/1e-4 | 0.9 | — | — | 19 | 46.25 | 54.15 | 27.67 | 78.34 | 88.0 | 546 s |
| `h4_crf_s42` | deberta-v3-large | 42 | 30 | 2e-5/1e-4 | 0.9 | **yes** | — | 11 | 46.67 | 54.25 | 31.26 | 79.03 | 89.9 | 898 s |
| `h8_span_s42` | deberta-v3-large | 42 | 30 | 2e-5/1e-4 | 0.9 | — | **yes** | 11 | 46.53 | 52.64 | 27.97 | 78.54 | 90.5 | 548 s |
| `final_s42` | deberta-v3-large | 42 | 30 | 1e-5/1e-4 | 1.0 | — | — | 19 | 47.40 | 54.71 | 29.68 | 76.72 | 89.2 | 541 s |
| `final_s13` | deberta-v3-large | 13 | 30 | 1e-5/1e-4 | 1.0 | — | — | 19 | 48.10 | 57.31 | 28.86 | 78.99 | 88.6 | 549 s |
| `final_s101` | deberta-v3-large | 101 | 30 | 1e-5/1e-4 | 1.0 | — | — | 26 | 46.87 | 55.78 | 28.42 | 78.68 | 89.2 | 552 s |

¹ `h1_deberta_s42` ran **before** calibrated checkpoint selection was
implemented, so its selection-time score is an **argmax** score and is not on the
same scale as the other rows. Its re-scored value is in §5.

---

## 4. The decisive diagnostic: H1 and the calibration failure

### 4.1 H1 dev learning curve (seed 42, 40 epochs, argmax decoding)

Dev Arg-C IoU F1 by epoch:
`ep1 2.51 · ep2 12.37 · ep3 19.65 · ep4 25.48 · ep5 29.80 · ep6 32.08 ·
ep7 34.98 · ep8 29.73 · ep9 34.30 · ep10 34.44 · ep11 34.70 · ep12 34.00 ·
ep13 35.89 · ep14 37.19 · ep15 35.33 · ep16 33.69 · ep17 37.89 · ep18 37.42 ·
ep19 36.94 · ep20 38.88 (best) · ep21 36.54 · ep22 37.55 · ep25 37.77 ·
ep30 37.75 · ep33 37.93 · ep35 37.82 · ep40 37.91`

Training loss reaches 0.045 by epoch 20 and 0.002 by epoch 40 — the model fully
memorises 1278 windows. Best epoch is consistently 17–20.

### 4.2 Error taxonomy (dev)

⚠ **Attribution caveat, stated so the paper does not mis-cite it.** The taxonomy
block below was computed on the dev predictions saved at **epoch 17**
(Arg-C 37.89); the per-role table in §4.3 was computed on the file after it was
overwritten at **epoch 20** (Arg-C 38.88). They are adjacent checkpoints of the
same run, not the same checkpoint. **For the paper, use the test taxonomy in
§6.4**, which is a single clean frozen checkpoint.

Over 497 gold scored arguments (dev):

| bucket | count | share of gold |
|---|---|---|
| correct | 250 | 50.3 % |
| missed (no overlapping prediction) | 86 | 17.3 % |
| killed by wrong window type | 56 | 11.3 % |
| role confusion (span right, role wrong) | 53 | 10.7 % |
| boundary (role right, IoU ≤ 0.5) | 52 | 10.5 % |
| **spurious** | **388** | — |

Window-type accuracy 89.24 % (141/158).

Recall by gold span length: 1–4 tok (n=82) **34.1**; 5–9 (n=121) 41.3;
10–19 (n=200) 56.0; 20–34 (n=84) 64.3; 35+ (n=10) 60.0.
Length bias on matched spans: mean **−1.00** tokens, median **+0.0**.

**The decisive number: 784 predicted spans against 497 gold.** Precision was
uniformly ≈30 across every role while recall was 50–65. That is a calibration
failure, not a representation failure.

### 4.3 Per-role dev (epoch-20 checkpoint, argmax decoding)

| role | P | R | F1 |
|---|---|---|---|
| Challenge | 43.48 | 66.67 | 52.63 |
| Results | 41.76 | 65.52 | 51.01 |
| Implications | 32.65 | 51.61 | 40.00 |
| Method | 29.67 | 50.47 | 37.37 |
| Context | 25.53 | 37.74 | 30.46 |
| Purpose | 22.45 | 44.00 | 29.73 |
| Analysis | 11.11 | 15.38 | 12.90 |
| Ethical | 0.00 | 0.00 | 0.00 |

Per-domain dev Arg-C IoU F1: ACL 47.79, cscw 44.07, bioinfo 43.90, jmir 36.81,
**dh 29.39**. Per-event-type: Results/Findings 54.12, Background/Introduction
36.95, Methods/Approach 34.06, Conclusions/Implications 33.33.

Top role confusions (gold → predicted): Method→Context 7, Context→Method 6,
Results→Method 6, Context→Results 5, Challenge→Results 5, Context→Challenge 4,
Purpose→Implications 3, Implications→Results 3, Results→Implications 3,
Analysis→Method 3.

---

## 5. Hypotheses tested

Reference targets, verified against the paper: **Arg-C IoU 41.61**,
**Arg-I IoU 53.57**, **trigger ROUGE-L 75.08**.

### 5.1 H5 — metric-aware decoding calibration: **KEPT**

Fine τ sweep on dev, `h1_deberta_s42`, **min_len = 3**:

| τ | Arg-C P | Arg-C R | **Arg-C F1** | Arg-I F1 | #pred (gold 497) |
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

Same sweep at **min_len = 2**: 0.50 → 43.99, 0.60 → 44.74, 0.70 → 46.49,
0.80 → 47.68, 0.85 → 47.65, 0.90 → 47.82, 0.93 → 46.67, 0.95 → 45.55,
0.97 → 41.98, 0.99 → 35.96.

**Gain: +9.12 Arg-C IoU on dev (38.88 → 48.00).** Precision 31.76 → 53.60 for a
recall cost of 6.64. The optimum is a **broad plateau over τ ∈ [0.70, 0.90]**
spanning ≈1 F1, collapsing only beyond 0.93 — the signature of a real
calibration correction rather than fitting dev noise.

**Per-role thresholds rejected.** Coordinate ascent over 9 extra per-role
thresholds gained only **+0.67** (47.05 → 47.72 at the then-current grid) on
158 dev windows. The 3-parameter global rule is used instead.

`min_len = 3` forfeits at most **7.6 %** of dev gold spans (those of length ≤ 2:
38/497; 5 of length 1, 33 of length 2) and is worth ≈1.5 precision points.

### 5.2 H2 — layer-wise learning-rate decay: **REJECTED** ⚠

`h3_llrd_s42`: lr 2e-5 encoder with LLRD 0.9, 30 epochs, dropout 0.15,
`w_type = 1.0`. Converged roughly twice as fast (dev Arg-C ≈44 by epoch 11 vs
epoch 20 for H1) but did not land higher.

⚠ **Confound, must not be reported as a clean ablation.** `h3_llrd_s42` was
tuned on the **narrower** τ grid (max 0.70), giving 46.58 at τ = 0.65, whereas
H1 was re-tuned on the **widened** grid (max 0.95), giving 48.00 at τ = 0.90.
`h3`'s widened-grid number was never measured, and its checkpoint has since been
deleted, so it cannot be recovered without retraining. **The −1.42 figure
previously quoted is not a like-for-like comparison.**

What *is* defensible: the two other LLRD-based runs (`h4`, `h8`) **were** tuned
on the widened grid and land at **46.52–46.80**, consistently ≈1.2–1.5 below
H1's 48.00. So "LLRD did not help" is supported; a precise per-point delta is
not. If the paper needs a clean number, retrain `h3` and re-tune on the widened
grid (≈10 min).

### 5.3 H4 — linear-chain CRF: **REJECTED** ⚠

`h4_crf_s42`: LLRD 0.9 + CRF on both BIO heads, Viterbi decoding, marginals via
forward–backward for confidences. 30 epochs, 898 s (≈30 s/epoch, ≈1.7× slower).

Widened-grid dev re-score: **Arg-C IoU 46.52** (P 53.7 / R 41.0), Arg-I 54.50,
Arg-C EM 30.79, Arg-I EM 33.75, ROUGE-L 79.40, at τ = 0.80, min_len = 3.
Argmax (Viterbi) dev: Arg-C 35.82 (P 29.4 / R 45.9), Arg-I 44.46.

⚠ Its intended control is `h3` (LLRD, no CRF), and `h3`'s widened-grid number is
unavailable (§5.2). So the previously quoted "**CRF = −0.06**" is **not a valid
isolation**. What is measured: CRF + LLRD scores **46.52 vs plain H1's 48.00**,
and CRF's Viterbi argmax (35.82) is *lower* than H1's argmax (38.88).

**Mechanistic evidence against H4 that does not depend on the confound:** the
decoding tuner independently selects `merge_gap = 0` on every configuration.
CRF was hypothesised to suppress fragmentation across the zero-token gaps
between adjacent gold spans; if fragmentation were the dominant error, a
positive `merge_gap` would have been selected. It never was. Over-prediction of
low-confidence spans was the dominant error, and a scalar threshold addresses it
directly.

### 5.4 H8 — span-level role re-classifier: **REJECTED** (clean)

Motivation: the dev taxonomy attributed 53/497 gold arguments to "span right,
role wrong", and Arg-I exceeded Arg-C by ≈10 points. A BIO head labels each
*token*; nothing in it sees a whole span at once.

Head: `Linear(3H → H) → GELU → Dropout → Linear(H → 9)` over
`[h_start ; h_end ; mean(h)]` taken from the **type-conditioned** word states
`w_cond`. Trained on gold spans with equal weight (`w_span_role = 1.0`);
at inference it re-labels each decoded span (spans and boundaries unchanged).

**This is a clean within-checkpoint ablation** — same model, same grid, override
toggled:

| configuration | Arg-C IoU | Arg-I IoU | Arg-C EM | τ* | min_len* |
|---|---|---|---|---|---|
| `h8_span_s42`, override **off** | 46.56 | — | — | 0.80 | 4 |
| `h8_span_s42`, override **on** | **46.80** | 53.42 | 27.63 | 0.75 | 4 |

**Gain +0.24**, below the pre-registered 0.5 falsification threshold → rejected.
Argmax dev: 34.31 (off) → 34.79 (on); Arg-I 40.35 unchanged in both, as
predicted (the override cannot move Arg-I because it does not move spans).

**Why the prediction failed** — worth one sentence in the paper: once the
confidence threshold removes low-confidence spans, most "span right, role wrong"
cases go with them. The taxonomy was counting the *same* errors that calibration
already fixes.

### 5.5 H7 — multi-seed posterior ensembling: **REJECTED** (clean)

Per-token posteriors averaged across the three final seeds before decoding, then
tuned on dev with the widened grid:

| system | argmax Arg-C | tuned Arg-C IoU | Arg-I IoU |
|---|---|---|---|
| `final_s42` alone | 36.69 | 47.66 | 55.50 |
| `final_s13` alone | 38.52 | **48.10** | 57.31 |
| `final_s101` alone | 37.32 | 46.99 | 54.64 |
| **3-seed posterior average** | **39.35** | 47.10 | 54.11 |

Ensembling clearly **helps argmax** decoding (39.35 vs 36.69–38.52) but **hurts
calibrated** decoding (47.10, below the seed mean of 47.58 and well below the
best seed's 48.10). Averaging compresses the posterior distribution that the H5
threshold operates on, so the two techniques are antagonistic.

**Consequence for the paper: the shipped system is a single model**, and the
headline is the mean ± std over three seeds, not an ensemble.

### 5.6 H3 — explicit event-type head and conditioning: **KEPT**

Oracle diagnostic on `h1_deberta_s42`, dev, τ = 0.7 / min_len = 3 — the same
argument spans re-scored with the gold event type substituted:

| window type source | Arg-C IoU | Arg-I IoU |
|---|---|---|
| model prediction (89.24 % accurate) | 47.05 | 56.89 |
| **oracle gold type** | **51.77** | **62.80** |

**Imperfect window typing costs 4.72 Arg-C / 5.91 Arg-I.** Kept.

### 5.7 H6 — rare-role rebalancing: **NOT RUN**

Analysis (67 train instances), Contradictions (2) and Ethical (1) together are
under 2 % of gold. Recorded as low expected information gain, not as a finding.

### 5.8 H2b — backbone sweep: **NOT RUN**

`answerdotai/ModernBERT-large` was downloaded and a config written
(`configs/h2_modernbert.json`) but never trained: with three architectural
hypotheses already falsified on the same backbone, a backbone swap was the
lowest-information remaining experiment. Stated as future work, not as a result.

### 5.9 Ablation summary (dev, seed 42)

| run | recipe | tuning grid | Arg-C IoU | Arg-I IoU | verdict |
|---|---|---|---|---|---|
| **`h1_deberta_s42`** | **plain** | widened (τ ≤ 0.95) | **48.00** | **55.56** | **kept** |
| `h8_span_s42` (override on) | +LLRD +span head | widened | 46.80 | 53.42 | H8 rejected |
| `h8_span_s42` (override off) | +LLRD +span head, unused | widened | 46.56 | — | — |
| `h4_crf_s42` | +LLRD +CRF | widened | 46.52 | 54.50 | H4 rejected ⚠ |
| `h3_llrd_s42` | +LLRD | **narrow (τ ≤ 0.70)** ⚠ | 46.58 | 54.60 | LLRD rejected ⚠ |
| 3-seed posterior ensemble | H7 | widened | 47.10 | 54.11 | H7 rejected |

Only the H8 (within-checkpoint) and H7 (same-grid) comparisons are clean
isolations. The h3/h4 rows share the grid confound of §5.2.

---

## 6. Results

### 6.1 Frozen protocol actually executed

1. Architecture and hyper-parameters fixed (§2), three seeds trained on **train**.
2. Checkpoints selected on **dev** (§2.9).
3. Decoding rule frozen on **dev** across seeds → τ = 0.90, min_len = 3,
   merge_gap = 0; dev Arg-C IoU **47.14 ± 0.26** (§2.10).
4. **Test evaluated once per seed** with that single rule. Nothing changed after.

Frozen-rule grid on dev (mean across the three seeds), min_len = 3:

| τ | s42 | s13 | s101 | mean | std |
|---|---|---|---|---|---|
| 0.50 | 43.52 | 45.67 | 44.55 | 44.58 | 1.07 |
| 0.60 | 44.06 | 46.05 | 44.51 | 44.87 | 1.04 |
| 0.70 | 44.73 | 46.69 | 44.87 | 45.43 | 1.09 |
| 0.80 | 45.98 | 48.10 | 45.28 | 46.45 | 1.47 |
| 0.85 | 46.62 | 47.60 | 45.95 | 46.73 | 0.83 |
| 0.88 | 46.91 | 47.05 | 46.58 | 46.85 | 0.24 |
| **0.90** | 47.40 | 47.15 | 46.87 | **47.14** | **0.26** |
| 0.92 | 47.34 | 46.94 | 45.84 | 46.71 | 0.78 |
| 0.95 | 45.82 | 44.44 | 44.82 | 45.03 | 0.71 |

### 6.2 Headline — Table 4 comparison (IoU argument extraction, test)

| Method | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| **OneIE** (previous best) | 51.11 | 56.29 | **53.57** | 39.69 | 43.71 | **41.61** |
| GPT (0-shot) | 43.03 | 55.56 | 48.50 | 30.40 | 39.25 | 34.26 |
| GPT (1-shot) | 50.14 | 50.22 | 50.18 | 34.60 | 34.66 | 34.63 |
| GPT (2-shot) | 49.12 | 51.29 | 50.18 | 33.99 | 35.49 | 34.72 |
| GPT (5-shot) | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| Qwen (5-shot) | 46.94 | 31.36 | 37.60 | 21.67 | 14.48 | 17.36 |
| Llama (1-shot) | 44.70 | 34.08 | 38.68 | 18.93 | 14.44 | 16.38 |
| DS-R1-Llama (1-shot) | 42.62 | 17.67 | 24.98 | 19.59 | 8.12 | 11.48 |
| **CARVE (ours)** | **63.46** | 53.35 | **57.95** | **55.30** | **46.47** | **50.48** |
| *± std over 3 seeds* | ±2.16 | ±2.98 | ±2.46 | ±1.13 | ±1.74 | ±1.15 |
| *worst seed* | 61.86 | 49.91 | 55.24 | 54.27 | 44.47 | 49.22 |
| **Δ vs OneIE** | **+12.35** | **−2.94** | **+4.38** | **+15.61** | **+2.76** | **+8.87** |

### 6.3 Headline — Table 3 comparison (trigger ROUGE-L, test)

| Method | P | R | **F1** |
|---|---|---|---|
| EEQA | 81.93 | 34.57 | 45.05 |
| DEGREE | 64.56 | 63.49 | 56.85 |
| OneIE | 73.73 | 79.40 | 72.40 |
| GPT (0-shot) | 65.38 | 72.73 | 67.57 |
| GPT (1-shot) | 72.67 | 77.77 | 74.05 |
| GPT (2-shot) | 73.38 | 78.45 | 74.76 |
| **GPT (5-shot)** (previous best) | 73.70 | 78.82 | **75.08** |
| Qwen (2-shot) | 57.27 | 69.71 | 61.18 |
| Llama (0-shot) | 54.88 | 61.07 | 55.83 |
| DS-R1-Llama (1-shot) | 41.81 | 41.94 | 40.72 |
| **CARVE (ours)** | **83.85** | 76.28 | **76.93 ± 0.80** |
| **Δ vs GPT 5-shot** | **+10.15** | **−2.54** | **+1.85** |

### 6.4 All four matching modes (test, mean ± std over 3 seeds)

| mode | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|
| Exact Match | 40.05 ±1.18 | 33.65 ±1.21 | 36.56 ±0.95 | 36.63 ±1.10 | 30.77 ±0.99 | 33.43 ±0.79 |
| Simple overlap | 75.47 ±2.27 | 63.41 ±2.78 | 68.90 ±2.21 | 64.75 ±1.13 | 54.41 ±1.99 | 59.11 ±1.22 |
| SciREX > 0.5 | 70.61 ±1.41 | 59.35 ±2.87 | 64.47 ±2.06 | 60.27 ±0.64 | 50.66 ±2.27 | 55.03 ±1.46 |
| **IoU > 0.5** | 63.46 ±2.16 | 53.35 ±2.98 | 57.95 ±2.46 | 55.30 ±1.13 | 46.47 ±1.74 | 50.48 ±1.15 |

### 6.5 Per-seed test results (frozen rule τ = 0.90, min_len = 3)

| | seed 42 | seed 13 | seed 101 |
|---|---|---|---|
| Arg-C IoU P / R / F1 | 55.12 / 44.47 / **49.22** | 56.50 / 47.28 / **51.48** | 54.27 / 47.65 / **50.75** |
| Arg-I IoU P / R / F1 | 61.86 / 49.91 / 55.24 | 65.92 / 55.16 / 60.06 | 62.61 / 54.97 / 58.54 |
| Arg-C EM P / R / F1 | 36.74 / 29.64 / 32.81 | 37.67 / 31.52 / 34.32 | 35.47 / 31.14 / 33.17 |
| Arg-I EM P / R / F1 | 40.00 / 32.27 / 35.72 | 41.26 / 34.52 / 37.59 | 38.89 / 34.15 / 36.36 |
| Arg-C overlap F1 | 57.74 | 60.06 | 59.54 |
| Arg-I overlap F1 | 66.67 | 71.09 | 68.93 |
| Arg-C SciREX F1 | 53.37 | 55.57 | 56.14 |
| Arg-I SciREX F1 | 62.10 | 65.78 | 65.53 |
| ROUGE-L P / R / F1 | 82.42 / 76.38 / 76.02 | 84.68 / 76.77 / 77.52 | 84.46 / 75.69 / 77.25 |
| **uncalibrated** Arg-C IoU | 39.08 (P 31.7 / R 50.8) | 41.44 (P 34.1 / R 52.9) | 39.15 (P 31.5 / R 51.8) |
| **uncalibrated** Arg-I IoU | 45.85 | 49.38 | 47.94 |

Seed 13 match counts (from the upstream CLI): Arg-I IoU 294/446 pred, 294/533
gold; Arg-C IoU 252/446, 252/533; Arg-I EM 184/446; Arg-C EM 168/446.

### 6.6 **The honest decomposition** — put this in the paper

Neither half of the method beats OneIE alone. Test, mean over 3 seeds:

| stage | Arg-C P | Arg-C R | **Arg-C F1** | Arg-I F1 |
|---|---|---|---|---|
| OneIE (previous best) | 39.69 | 43.71 | **41.61** | 53.57 |
| ours, **uncalibrated** argmax BIO | 32.43 ±1.45 | **51.83 ±1.05** | **39.89 ±1.34** | 47.72 ±1.77 |
| ours, **+ calibrated decoding** | 55.30 | 46.47 | **50.48** | 57.95 |

1. **Uncalibrated, the reframing scores below OneIE** (39.89 vs 41.61, −1.72).
   Do not claim span segmentation alone beats the baselines.
2. **What the reframing buys is recall: 51.83 vs 43.71 (+8.12).** The
   clause-level tagger genuinely finds more gold arguments than graph decoding,
   templated generation or span-QA. It pays with precision (32.43).
3. **Calibration converts that recall headroom into +10.59 F1.** A threshold
   cannot *create* recall; it can only trade it. A baseline without the recall
   headroom could not reach 50.48 by thresholding.

Corresponding dev figures: 38.88 → 48.00.
Trigger ROUGE-L is **identical** calibrated and uncalibrated (76.93), because
the AAO head is not thresholded.

### 6.7 Test error taxonomy (seed 13, frozen) — **use this one**

Over 533 gold scored arguments:

| bucket | count | share of gold |
|---|---|---|
| correct | 246 | 46.2 % |
| **missed** | **177** | **33.2 %** |
| boundary (role right, IoU ≤ 0.5) | 44 | 8.3 % |
| role confusion (span right, role wrong) | 41 | 7.7 % |
| killed by wrong window type | 25 | 4.7 % |
| spurious | 102 | — |

Window-type accuracy **91.41 % (149/163)**; most frequent confusion
Results/Findings → Conclusions/Implications (6 windows).

**Recall by gold span length (test, seed 13):**

| span length | n | recall @ IoU > 0.5 |
|---|---|---|
| 1–4 | 77 | **14.3** |
| 5–9 | 141 | 34.8 |
| 10–19 | 202 | 58.4 |
| 20–34 | 106 | 61.3 |
| 35+ | 7 | 42.9 |

Length bias on matched spans: mean **−1.00** tokens, median **+0.0** — boundaries
are unbiased; the short-span collapse is caused by the frozen `min_len = 3` and
τ = 0.90. (For reference, 6.8 % of test gold scored spans are ≤2 tokens: 11 of
length 1, 25 of length 2.)

Top role confusions (test, seed 13; gold → predicted): Results→Method 5,
Method→Context 5, Purpose→Method 4, Challenge→Implications 4, Context→Challenge 3,
Context→Method 3, Purpose→Context 2, Challenge→Results 2, Method→Results 2,
Analysis→Results 2, Analysis→Context 2, Results→Context 1.

### 6.8 Per-domain (test, mean ± std over 3 seeds)

| domain | Arg-I IoU F1 | Arg-C IoU F1 | gold scored args |
|---|---|---|---|
| bioinfo | 78.14 ± 2.69 | **74.39 ± 1.88** | 52 |
| ACL | 66.24 ± 5.96 | 59.21 ± 4.79 | 60 |
| jmir | 57.50 ± 1.34 | 52.11 ± 2.01 | 218 |
| cscw | 60.16 ± 6.12 | 48.61 ± 2.98 | 91 |
| **dh** | 40.14 ± 1.65 | **29.75 ± 1.11** | 112 |

Digital Humanities is 2.5× worse than computational biology and stable across
seeds. This reproduces the paper's humanities finding; our method does **not**
fix it.

### 6.9 Per-event-type (test, mean ± std)

| event type | Arg-I IoU F1 | Arg-C IoU F1 | gold |
|---|---|---|---|
| Results/Findings | 64.31 ± 1.17 | 57.07 ± 1.43 | 132 |
| Methods/Approach | 60.48 ± 2.33 | 51.89 ± 0.99 | 220 |
| Background/Introduction | 51.60 ± 3.88 | 46.08 ± 2.12 | 143 |
| Conclusions/Implications | 43.65 ± 5.77 | **34.11 ± 4.39** | 38 |

### 6.10 Per-role (test, mean over 3 seeds, Arg-C IoU)

| role | P | R | F1 | gold |
|---|---|---|---|---|
| Challenge | 68.19 | 56.32 | **61.67** | 58 |
| Method | 64.69 | 58.97 | **61.63** | 143 |
| Results | 60.71 | 62.36 | **61.50** | 116 |
| Purpose | 56.52 | 30.23 | 39.39 | 43 |
| Implications | 58.42 | 28.07 | 37.28 | 19 |
| Context | 34.14 | 28.17 | **30.86** | 142 |
| Analysis | 0.00 | 0.00 | **0.00** | 10 |
| Contradictions | 0.00 | 0.00 | 0.00 | 1 |
| Ethical | 0.00 | 0.00 | 0.00 | 1 |

Context is simultaneously the second-most frequent role and among the weakest.
The three tail roles score exactly zero.

---

## 7. Audit — leakage and protocol

### 7.1 Official-CLI re-verification

```
python3 third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py \
  --pred preds/test_preds_s13.jsonl \
  --gold third_party/SciEvent/SciEvent_data/ONEIE/all_splits/test.oneie.json
```
```
[ROUGE-L OVERALL] P: 84.68%, R: 76.77%, F1: 77.52%
Argument Identification (Exact)  - P: 41.26 (184/446)  R: 34.52 (184/533)  F1: 37.59
Argument Classification (Exact)  - P: 37.67 (168/446)  R: 31.52 (168/533)  F1: 34.32
Argument Identification (IoU)    - P: 65.92 (294/446)  R: 55.16 (294/533)  F1: 60.06
Argument Classification (IoU)    - P: 56.50 (252/446)  R: 47.28 (252/533)  F1: 51.48
```
Identical to §6.5 seed 13 to the last decimal.

### 7.2 Leakage checks — all clean

| check | result |
|---|---|
| train ∩ test windows | **0** |
| train ∩ dev windows | **0** |
| dev ∩ test windows | **0** |
| exact window-text duplicates train → test | **0 / 163** |
| `load_split` calls in training path | only `"train"` and `eval_split` |
| `eval_split` in all 6 configs | unset → defaults to `dev` |
| frozen rule written at | 21:35:50 |
| first test prediction written at | 21:36:05 (rule predates it) |
| gold event type used at inference | **no** |
| predicted windows vs gold | 163 / 163, ids match exactly |
| predicted scored args vs gold | **446 / 533 = 0.837** (we *under*-predict) |
| duplicate `(span, role)` predictions | **0** |
| `third_party/SciEvent` modified files | **0** |

The duplicate check matters because `compute_f1` accumulates matches into a
`set` of `(pred, gold)` pairs; repeated identical spans could interact with that.
We emit none. The under-prediction ratio rules out winning precision by flooding.

### 7.3 Benchmark finding: the released split is window-level, not document-level

The paper describes an 80/10/10 split **by document**. The released
`data_scripts/shared/split_data.py` dedups and splits on **`wnd_id`**
(`key = item["wnd_id"]`), stratified by event type. Consequence:

| | value |
|---|---|
| test documents | 147 |
| test documents also appearing in train | **143 (97.3 %)** |
| test documents unseen in train *and* dev | 2 |
| dev documents also appearing in train | 128 / 133 |

Windows are disjoint; abstracts are not. **This applies identically to OneIE,
DEGREE and EEQA**, which the benchmark README trains and scores on these exact
files — so the Table 3 / Table 4 comparison remains apples-to-apples. It must
nevertheless be disclosed.

### 7.4 Document-disjoint robustness probe

`scripts/make_docsplit.py` rebuilds a split keyed on `doc_id`, stratified by each
document's event-type signature, seed 42. Verified zero document overlap.

| | train | dev | test |
|---|---|---|---|
| documents | 394 | 45 | 61 |
| windows | 1265 | 148 | 186 |
| Background / Methods / Results / Conclusions | 387/383/302/193 | 45/45/36/22 | 56/56/44/30 |
| ACL / bioinfo / cscw / dh / jmir | 255/227/273/243/267 | 24/30/46/28/20 | 30/28/44/56/28 |

Identical protocol: 3 seeds, rule **re-frozen on the new dev** → τ = 0.85,
min_len = 3, dev Arg-C **47.79 ± 1.64** (per-seed 45.92 / 48.95 / 48.51), then
one test evaluation.

| | Arg-I IoU F1 | **Arg-C IoU F1** | ROUGE-L F1 |
|---|---|---|---|
| ours, official split | 57.95 ± 2.46 | **50.48 ± 1.15** | 76.93 ± 0.80 |
| ours, **document-disjoint** | 52.27 ± 1.43 | **44.93 ± 0.95** | 76.20 ± 1.00 |
| worst seed (doc-disjoint) | 50.66 | 44.18 | — |
| paper best (official split only) | 53.57 | 41.61 | 75.08 |

Document-disjoint P/R: Arg-I 57.51 ± 3.13 / 47.95 ± 0.82; Arg-C 49.41 ± 1.71 /
41.24 ± 1.44; Exact Arg-I 31.91/26.62/29.01, Arg-C 28.34/23.65/25.77;
ROUGE-L P 83.40 ± 2.19 / R 75.91 ± 1.42.

**Decomposing the 5.55-point drop.** The re-split is harder by construction: its
test half is **33.7 %** DH gold arguments versus **21.0 %** officially.
Domain by domain:

| domain | official F1 (n) | doc-disjoint F1 (n) | Δ |
|---|---|---|---|
| ACL | 59.21 (60) | 51.10 (63) | −8.11 |
| bioinfo | 74.39 (52) | 52.96 (57) | −21.43 |
| cscw | 48.61 (91) | 51.35 (86) | **+2.75** |
| dh | 29.75 (112) | 25.88 (189) | −3.87 |
| jmir | 52.11 (218) | 52.97 (166) | **+0.86** |

Reweighting the doc-disjoint per-domain scores to the official domain mix:

| | Arg-C IoU F1 |
|---|---|
| doc-disjoint, its own DH-heavy mix | 43.38 |
| doc-disjoint, **reweighted to official mix** | **46.79** |
| official split, same weighting | 49.79 |

**≈3.4 points of the drop come from the DH-heavy test mix; ≈3.0 points from
removing document overlap.** Two domains improved under the stricter split,
which is not what systematic leakage looks like.

⚠ **Asymmetry to state explicitly:** OneIE was never re-measured on the
document-disjoint split and would very likely drop too. Comparing our 44.93
against OneIE's official-split 41.61 is therefore *unfavourable to us*. The
headline claim rests on the official split, where the comparison is genuinely
like-for-like.

### 7.5 Complete disclosure of every contact with test data

1. `tests/test_contract.py` — uses test **gold labels** to verify the exporter
   scores 100.00. No model involved, hence no model-selection signal.
2. A one-off check of the **gold span-length distribution** on test (6.8 % ≤2
   tokens), run to sanity-check whether `min_len` would transfer. `min_len = 3`
   had **already** been selected by the dev grid before this was computed, and
   the final value came from `freeze_rules.py` on dev alone. It changed nothing,
   but it was looked at and is disclosed rather than omitted.
3. A tokenizer length check on test (max 240 sub-tokens) — token counts only.
4. `scripts/make_docsplit.py` pools all three official splits to build the
   document-disjoint re-split. That re-split is therefore *not* comparable to the
   paper and is reported only as a robustness probe.
5. The single frozen evaluation itself (§6).

### 7.6 Is the threshold "gaming the metric"?

τ, `min_len`, `merge_gap` are three scalars fitted on dev and frozen before test
— the same class of choice as a learning rate or a checkpoint criterion.
Supporting evidence that it is not dev-overfitting: the dev optimum is a broad
plateau (§5.1); the rule was chosen to maximise the **mean across seeds**
(§2.10); dev std at the chosen point is **0.26**; and **test (50.48) came out
above dev (47.14)**, the opposite of dev-overfitting.

What would be illegitimate and was **not** done: tuning τ on test, selecting
checkpoints on test, or reporting the best of several test evaluations.

---

## 8. Limitations to state in the paper

### 8.1 Ranked remaining weaknesses

1. **Short spans.** Recall at IoU > 0.5 is **14.3 %** for 1–4-token gold spans and
   34.8 % for 5–9, against ≈60 % for 10–34. Self-inflicted by `min_len = 3` and
   τ = 0.90. A **length-conditioned threshold** should recover much of it.
2. **Digital Humanities**: 29.75 vs 74.39 for bioinfo, stable across seeds.
3. **Tail roles at zero**: Analysis (10 test instances), Contradictions (1),
   Ethical (1) all score 0.00.
4. **Conclusions/Implications** is the weakest event type (34.11), and the
   Implications role is weak (37.28).
5. **Window typing still costs 4.7 %** of gold arguments at 91.41 % accuracy.
6. **Exact match is low** (Arg-C EM 33.43 vs IoU 50.48): spans are approximately
   right, not exactly right.

### 8.2 The operating point

We beat OneIE on Arg-C precision **and** recall, but on **Arg-I our recall is
2.94 lower** than OneIE's (53.35 vs 56.29) while precision is 12.35 higher; the
same shape appears in ROUGE-L (recall −2.54, precision +10.15). OneIE is reported
at its own natural operating point and we at a tuned one. A precision/recall
matched comparison would narrow the Arg-C gap. A recall-oriented operating point
exists as a single dial but was **not** evaluated on test, because the rule was
frozen first.

### 8.3 The segment ordinal, deliberately unused

The window id encodes the segment's ordinal position within its abstract. A
majority-class rule on that ordinal alone predicts the window event type at
**98.73 %** on dev, versus **89.24 %** (dev) / **91.41 %** (test) from text.
Per-ordinal majority on train: index 0 → Background/Introduction 97.5 % (n=400);
index 1 → Methods/Approach 95.7 % (n=394); index 2 → Results/Findings 88.5 %
(n=330); index 3 → Conclusions/Implications 100.0 % (n=154).

Oracle typing is worth **+4.72 Arg-C / +5.91 Arg-I** (§5.6), so most of that
headroom is realisable. It is **excluded from the headline system** because the
paper's baselines did not use it, and reported only as measured headroom. A
deployed pipeline would legitimately have this feature.

### 8.4 Scope

- **Segmentation is not modelled**; gold windows are consumed, as all three tuned
  baselines do. No segmentation number is reported.
- Documents per abstract: 3 segments → 224 docs, 2 → 133, 4 → 119, 1 → 22
  (train+dev), so most abstracts contribute 2–4 windows.

---

## 9. Corrections to earlier reporting

Recorded so nothing incorrect reaches a paper draft.

1. **The LLRD ablation (§5.2) and the CRF isolation (§5.3) are grid-confounded.**
   `h3_llrd_s42` was tuned on the narrower τ grid (max 0.70) while H1, h4 and h8
   were tuned on the widened grid (max 0.95). The previously quoted figures
   "LLRD −1.42" and "CRF −0.06 vs its LLRD-matched control" are **not
   like-for-like** and must not be printed as clean ablations. `h3`'s checkpoint
   was deleted to free disk, so recovering the number requires a ≈10-minute
   retrain. The qualitative conclusions (neither helps) remain supported by the
   widened-grid runs h4 = 46.52 and h8 = 46.56/46.80 against H1 = 48.00.
2. **The dev error taxonomy and the dev per-role table come from different
   epochs** of the same run (17 and 20). Use the test taxonomy in §6.7 for the
   paper.
3. `h1_deberta_s42`'s registry score (38.88) is an **argmax** selection-time
   score, on a different scale from the other registry rows, which use calibrated
   selection. Do not tabulate them together without that note.

---

## 10. Reproduction

```bash
cd CARVE
pip install -r requirements.txt
bash scripts/reproduce.sh      # tests → 3 seeds → freeze on dev → test once → analysis
bash scripts/audit.sh          # §7 leakage and protocol audit
python3 scripts/results_tables.py # regenerates §6.2–6.4 and §6.8–6.10
```

Artefacts: `assets/decoding_rules.json` (frozen rule),
`assets/decoding_rules.docsplit.json`, `preds/test_preds_s{42,13,101}.jsonl`,
`preds/docsplit_test_s{42,13,101}.jsonl`,
`runs/<run>/log.json` (per-epoch history), `docs/PAPER_NOTES.md`.

Total compute for everything in this document: ≈2.5 GPU-hours on one RTX 5880 Ada.

### 5.10 Combined ablation — both inert decisions removed at once

Both §5.6 (conditioning) and the two-head split are individually inert (§7.6 of
PAPER_VI). Neither combined config was trained before this note. Trained here:
`configs/carve_simple.json` (`single_head=true`, `use_type_cond=false`), 3 seeds,
scored with the exact same grid as Table 12 (tau ∈ {0.5,...,0.95}, min_len ∈ {2,3,4},
per-seed best, not a shared frozen rule — apples-to-apples with that table only).

| configuration | seed 42 | seed 13 | seed 101 | mean ± std | Δ vs full |
|---|---|---|---|---|---|
| Full model (CARVE) | 47.40 | 48.10 | 46.87\* | 47.58 ± 0.55 | — |
| − event-type conditioning only | — | — | — | 47.34 ± 1.61 | −0.24 |
| − two-head split only | 50.10 | 46.97 | 47.85 | 48.31 ± 1.62 | +0.75 |
| **− both, simultaneously (`carve_simple`)** | **49.19** | **47.82** | **48.48** | **48.50 ± 0.68** | **+0.92** |

(\* the "full model" per-seed numbers shown for §5.9/Table 12 use this script's
own wider grid, not the officially frozen single rule; see the discrepancy note
below.)

**Both single-factor deltas are individually inside seed noise (±1.6), and so is
the combined one (+0.92, std 0.68 — smaller variance than either single ablation,
because the two changes partly cancel each other's seed-to-seed swings, e.g.
seed 42's high single-head value 50.10 is not repeated here at 49.19).** The
combined config is not worse than the full model on any seed, and its mean is
the highest of the four rows, but the gap to the full model (0.92) is still
well under one seed-std (0.55–1.6) of every row in this table, so it is not a
demonstrated improvement — only a demonstration that removing both add-ons does
not hurt, which is what each ablation already showed separately. No test
evaluation was run for this configuration, consistent with every other row in
Table 12 (ablations are dev-only, not part of the frozen test protocol).

**Test evaluation of `carve_simple` (requested explicitly; one look, after the
rule was frozen on dev).** `scripts/freeze_and_test_simple.py`, same protocol as
the headline system: one (tau, min_len) maximising mean dev Arg-C over the three
seeds on the `freeze_rules.py` grid, merge_gap 0, applied to every seed.

Frozen rule: tau 0.90, min_len 3 (identical to CARVE's own) — dev 48.10 ± 0.96
(49.19 / 47.39 / 47.72) vs CARVE 47.14 ± 0.26.

| test | Arg-C IoU | Arg-I IoU | ROUGE-L |
|---|---|---|---|
| CARVE (two heads + conditioning) | 50.48 ± 1.15 | 57.95 ± 2.46 | 76.93 |
| **CARVE-simple** (one merged head, no conditioning) | **50.73 ± 0.19** (50.70 / 50.94 / 50.55) | **58.58 ± 0.57** | 77.22 ± 0.81 |

Paired window bootstrap (5,000 resamples, 3-seed means on both sides):
Arg-C Δ +0.25, 95 % CI [−1.66, +2.12], P(Δ>0) 0.60; Arg-I Δ +0.64,
CI [−1.27, +2.46]. **No difference.** The simpler configuration matches the
shipped one on test and has ~6× lower seed variance on Arg-C.
Predictions: `preds/test_carve_simple_s{42,13,101}.jsonl`.
