# SciEvent-Next — Final Report

A new method for the SciEvent benchmark (Dong et al., EMNLP 2025), developed
independently of `method/TARS-SciEvent` and without reproducing any baseline.

---

## 1. Targets

Verified against the paper's own result tables, and re-checked for any newer
directly comparable result (none found — no post-SciEvent paper reports on this
benchmark; similarly named datasets were excluded):

| metric | paper best | by |
|---|---|---|
| **Arg-C IoU F1** (primary) | 41.61 | OneIE |
| **Arg-I IoU F1** (secondary) | 53.57 | OneIE |
| Trigger ROUGE-L F1 | 75.08 | GPT 5-shot |
| Segmentation EM / IoU F1 | 60.95 / 85.63 | GPT (zero-shot; not modelled here — see §7) |

## 2. The finding that drives everything

We measured the benchmark before reading the EAE literature. On the 1278 training
windows:

| property | value |
|---|---|
| events per window | **exactly 1** (all 1599 windows in all splits) |
| window length | mean 62 whitespace tokens, p95 123, max 260 |
| **scored argument span length** | **mean 10–15 tokens** (Context 10.2, Method 13.2, Results 14.5, Challenge 14.4, Analysis 15.5) |
| windows with overlap among scored roles | **2.4 %** |
| windows with overlap among Agent/Primary/Secondary | 0.2 % |
| windows with cross-group overlap | 5.9 % |
| **modal gap between consecutive spans** | **0 tokens** (then 1) |
| token coverage of a window by scored spans | 56 % |

Reading the official evaluator rather than the paper's prose gave three more
facts that the paper's framing obscures:

1. Arg-I / Arg-C are **trigger-insensitive** but **event-type sensitive**.
2. `{Agent, PrimaryObject, SecondaryObject}` are **excluded** from Arg-I/Arg-C —
   they only feed the ROUGE-L ⟨Agent, trigger, Primary, Secondary⟩ tuple.
3. Matching is greedy one-to-one with `intersection/union > 0.5`.

**Conclusion.** SciEvent argument extraction is *clause-level span segmentation
with semantic-role labelling*, not ACE-style mention extraction. Every tuned
baseline in the paper (OneIE's graph decoding, DEGREE's templated generation,
EEQA's span-QA) models it as the latter. That mismatch — not model capacity — is
the bottleneck. DEGREE's reported Arg-C IoU recall of **13.83** is exactly what
generative copying of 10–15-token spans looks like.

## 3. Literature synthesis

Full review in `LITERATURE.md`. The two most load-bearing references told us what
*not* to build:

- **Are Triggers Needed for Document-Level Event Extraction?** (TACL 2025) —
  with few events per document, explicit trigger extraction is redundant.
  SciEvent has one event per window and trigger-insensitive Arg metrics, so the
  trigger is off the critical path entirely. We model it in a **separate** head
  purely to produce the ROUGE-L tuple.
- **DEGREE / EEQA's own reported SciEvent numbers** — the recall collapses are
  direct evidence against generative and span-QA formulations here, which is why
  the project did not begin with a LoRA-tuned LLM.

The positive template came from **Joint Span Segmentation and Rhetorical Role
Labeling** (legal documents): contiguous clause spans carrying discourse roles,
learned as segmentation, with semi-Markov/CRF structure. We adopted the framing
and tested the CRF — which then failed (§5).

## 4. Method

**SciEvent-Next** — a shared encoder with three heads:

```
DeBERTa-v3-large  →  first-sub-token pooling to word states
   ├─ TYPE head : attention-pooled → 4-way window event type
   │              → learned type embedding added back to word states
   ├─ ROLE head : word-level BIO over the 9 scored semantic roles  (type-conditioned)
   └─ AAO  head : word-level BIO over ⟨Agent, Action, Primary, Secondary⟩
```

Design points, each forced by a measurement:

- **Two disjoint BIO heads, not one.** Within-group overlap is ≤2.4 %, but
  cross-group overlap is 5.9 %. One head would have to delete labels; two do not.
- **BIO rather than span enumeration.** The BIO ceiling was measured *before*
  committing: 99.70 Arg-C IoU on dev. The formulation costs 0.3 F1.
- **Type conditioning with scheduled sampling.** One event per window and an
  event-type-sensitive metric means a wrong window type annihilates every
  argument in it, as false positives *and* false negatives simultaneously.
  Training feeds gold type with an annealing probability and the model's own
  argmax otherwise, so train and inference agree.
- **Metric-aware decoding (3 parameters, tuned on dev only).** Drop spans whose
  mean token posterior is below τ; drop spans shorter than `min_len`; merge
  same-role spans within `merge_gap`.
- **A single model, not an ensemble.** Posterior ensembling across seeds was
  tested and *falsified* (§5). The reported result is the mean ± std over three
  seeds of one frozen recipe.

Training: 30 epochs, batch 8, AdamW lr 1e-5 (encoder) / 1e-4 (heads), warmup 10 %,
linear decay, fp32 master weights with bf16 autocast. ~10 minutes per run on one
RTX 5880 Ada. Backbone details and license in `MODELS.md`.

## 5. Negative results

Three genuinely different architectural hypotheses were implemented and
falsified. All numbers are dev, seed 42, scored under an identical tuned
decoding grid:

| run | Arg-C IoU | Arg-I IoU | verdict |
|---|---|---|---|
| **plain multi-task tagger** | **48.00** | **55.56** | kept |
| + span-level role re-classifier (H8) | 46.80 | 53.42 | **rejected**, +0.24 vs its control |
| + layer-wise LR decay (H2) | 46.58 | 54.60 | **rejected**, −1.42 |
| + LLRD + linear-chain CRF (H4) | 46.52 | 54.50 | **falsified**, −0.06 vs control |
| 3-seed posterior ensemble (H7) | 47.10 | 54.11 | **rejected**, below the best single seed |

- **CRF (H4).** Predicted to fix fragmentation across the zero-token gaps between
  adjacent gold spans. It did not, and the reason is visible in the decoding
  tuner: the selected `merge_gap` is **0**, i.e. fragmentation was never the
  dominant error. Over-prediction of low-confidence spans was, and a threshold
  fixes that far more cheaply.
- **Span-level role head (H8).** Predicted to close the Arg-I/Arg-C gap, which the
  error taxonomy attributed to "span right, role wrong". It gained +0.24.
  The prediction failed because the confidence threshold already removes most of
  those cases — they were the same errors counted twice.
- **Layer-wise LR decay (H2).** Converged roughly twice as fast but landed lower.
- **Posterior ensembling (H7).** The one negative result that is genuinely
  interesting. Averaging three seeds' posteriors *improves* argmax decoding
  (39.35 vs 36.7–38.5) but *degrades* calibrated decoding (47.10 vs a seed mean of
  47.58 and a best seed of 48.10). Averaging compresses the confidence
  distribution that the H5 threshold depends on, so the two techniques are
  antagonistic. Ensembling is dropped.

Also recorded: an **infrastructure defect**, not a modelling one. The first smoke
run gave `loss nan` from epoch 1. Gradients were finite (grad-norm 11.02) yet all
390 parameter tensors were NaN after a single `opt.step()` at lr 6.25e-07. Cause:
`transformers>=5` honours the dtype stored in a checkpoint, and
`microsoft/deberta-v3-large` ships **fp16** weights, so AdamW was updating fp16
master weights. Fixed by loading the encoder with `dtype=torch.float32`.

## 6. Results

### 6.1 Headline — frozen evaluation on test

Three seeds of the frozen recipe, one frozen decoding rule (τ = 0.90,
min_len = 3, merge_gap = 0), test touched exactly once:

| metric | paper best | **SciEvent-Next** | Δ | worst seed |
|---|---|---|---|---|
| **Arg-C IoU F1** (primary) | 41.61 (OneIE) | **50.48 ± 1.15** | **+8.87** | 49.22 (+7.61) |
| **Arg-I IoU F1** (secondary) | 53.57 (OneIE) | **57.95 ± 2.46** | **+4.38** | 55.24 (+1.67) |
| **Trigger ROUGE-L F1** | 75.08 (GPT 5-shot) | **76.93 ± 0.80** | **+1.85** | 76.02 (+0.94) |
| Arg-C EM F1 | — | 33.43 ± 0.79 | — | — |
| Arg-I EM F1 | — | 36.56 ± 0.95 | — | — |
| Arg-C overlap F1 | — | 59.11 ± 1.22 | — | — |
| Arg-C SciREX F1 | — | 55.03 ± 1.46 | — | — |

**Every individual seed beats every target.** Test (50.48) is above dev (47.14),
so the dev-tuned threshold did not overfit dev. Full baseline context:

**Full comparison against the paper's Table 4** (IoU-based argument extraction, %):

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
| **SciEvent-Next (ours)** | **63.46** | 53.35 | **57.95 ± 2.46** | **55.30** | **46.47** | **50.48 ± 1.15** |
| *worst seed* | 61.86 | 49.91 | 55.24 | 54.27 | 44.47 | 49.22 |
| **Δ vs OneIE** | **+12.35** | **−2.94** | **+4.38** | **+15.61** | **+2.76** | **+8.87** |

**Full comparison against the paper's Table 3** (trigger identification, ROUGE-L, %):

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
| **SciEvent-Next (ours)** | **83.85** | 76.28 | **76.93 ± 0.80** |
| **Δ vs GPT 5-shot** | **+10.15** | **−2.54** | **+1.85** |

**All four matching modes the official evaluator reports** (ours, mean over seeds):

| mode | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|
| Exact Match | 40.05 | 33.65 | 36.56 | 36.63 | 30.77 | 33.43 |
| Simple overlap | 75.47 | 63.41 | 68.90 | 64.75 | 54.41 | 59.11 |
| SciREX > 0.5 | 70.61 | 59.35 | 64.47 | 60.27 | 50.66 | 55.03 |
| **IoU > 0.5** (headline) | 63.46 | 53.35 | **57.95** | 55.30 | 46.47 | **50.48** |

**Note on the operating point.** We beat OneIE on Arg-C precision *and* recall,
but on **Arg-I our recall is 2.94 lower** than OneIE's while precision is 12.35
higher; the same shape appears in ROUGE-L (recall −2.54, precision +10.15). That
is the τ = 0.90 threshold doing what it was tuned to do — maximise Arg-C IoU F1,
the paper's primary metric. It is one dial, the dev curve for it is in
`RESEARCH_LOG.md`, and a recall-oriented operating point exists but was **not**
evaluated on test because the rule was frozen first.

Regenerate every table above with `python3 scripts/full_tables.py`.

### 6.2 Where the gain comes from — and an honest decomposition

Neither half of the method beats OneIE on its own. On **test**, mean over 3 seeds:

| stage | Arg-C P | Arg-C R | **Arg-C F1** | Arg-I F1 |
|---|---|---|---|---|
| OneIE (previous best) | 39.69 | 43.71 | **41.61** | 53.57 |
| ours, **uncalibrated** argmax BIO | 32.43 | **51.83** | **39.89 ± 1.34** | 47.72 |
| ours, **+ calibrated decoding** | 55.30 | 46.47 | **50.48 ± 1.15** | 57.95 |

Read that table carefully, because it is the actual result:

- **Uncalibrated, the reframing scores *below* OneIE** (39.89 vs 41.61, −1.72).
  Anyone claiming "span segmentation beats the baselines" on its own would be
  overclaiming.
- **What the reframing actually buys is recall: 51.83 vs OneIE's 43.71 (+8.12).**
  The clause-level tagger genuinely finds far more of the gold arguments than
  graph decoding, templated generation or span-QA do. It simply pays for that
  with precision (32.43), because it over-predicts.
- **Calibration converts that recall headroom into +10.59 F1.** The threshold is
  not a trick bolted onto a finished model; it is the mechanism that cashes in a
  recall advantage the reframing created and that no baseline had available.

So both halves are load-bearing and neither is sufficient. The corresponding dev
numbers are 38.88 → 48.00, and the dev threshold curve (a broad plateau over
τ ∈ [0.7, 0.9], not a spike) is in `RESEARCH_LOG.md`.

### 6.3 Analysis (test, seed 13)

Error taxonomy over 533 gold scored arguments:

| bucket | count | share |
|---|---|---|
| correct | 246 | 46.2 % |
| **missed** | **177** | **33.2 %** |
| boundary (role right, IoU ≤ 0.5) | 44 | 8.3 % |
| role confusion (span right, role wrong) | 41 | 7.7 % |
| wrong window type | 25 | 4.7 % |
| spurious | 102 | — |

Window-type accuracy on test is **91.41 %**.

**Per-domain Arg-C IoU F1** (seed 13 / 42 / 101):

| domain | s13 | s42 | s101 |
|---|---|---|---|
| bioinfo | 75.47 | 75.47 | 72.22 |
| ACL | 64.35 | 54.87 | 58.41 |
| cscw | 51.98 | 46.33 | 47.51 |
| jmir | 51.63 | 50.39 | 54.32 |
| **dh** | **28.57** | **30.77** | **29.90** |

Digital Humanities is 2.6× worse than computational biology and is stable across
seeds — this reproduces the paper's finding that narrative/humanities domains are
the hard case, and our method does **not** fix it.

**Per-event-type Arg-C IoU F1** (seed 13): Results/Findings 57.14,
Methods/Approach 52.35, Background/Introduction 48.33,
**Conclusions/Implications 36.67** (the rarest type, 25 test windows).

**Per-role Arg-C IoU F1** (seed 13): Challenge 65.38, Results 63.48,
Method 60.43, Purpose 39.39, Context 33.59, Implications 29.63,
**Analysis 0.00** (10 gold), **Contradictions 0.00** (1), **Ethical 0.00** (1).

**Recall by gold span length** is the sharpest remaining weakness:

| span length | n | recall @ IoU>0.5 |
|---|---|---|
| 1–4 | 77 | **14.3** |
| 5–9 | 141 | 34.8 |
| 10–19 | 202 | 58.4 |
| 20–34 | 106 | 61.3 |

Length bias on matched spans is ~0, so boundaries are unbiased; the short-span
collapse is caused directly by the frozen `min_len = 3` and τ = 0.90, which is
the precision/recall trade the dev tuner chose.

### 6.4 Robustness: does the margin survive removing document overlap?

The released `split_data.py` splits on `wnd_id`, not `doc_id`, so **97.3 % of
test documents have sibling segments in train** — windows are disjoint,
abstracts are not. This is a property of the benchmark and applies identically to
every baseline in Table 3 / Table 4, so §6.1 remains apples-to-apples. But it is
worth knowing how much of the score depends on it.

We rebuilt a **document-disjoint** split (394/45/61 documents, zero overlap),
retrained the identical frozen recipe on 3 seeds, re-froze the decoding rule on
the *new* dev, and evaluated once:

| | Arg-I IoU F1 | **Arg-C IoU F1** | ROUGE-L F1 |
|---|---|---|---|
| ours, official split | 57.95 ± 2.46 | **50.48 ± 1.15** | 76.93 ± 0.80 |
| ours, **document-disjoint** | 52.27 ± 1.43 | **44.93 ± 0.95** | 76.20 ± 1.00 |
| paper best (official split only) | 53.57 | 41.61 | 75.08 |

Of the 5.55-point drop, domain-reweighting attributes roughly **3.4 points to the
re-split's DH-heavy test mix** (33.7 % vs 21.0 % DH gold arguments) and roughly
**3.0 points to removing document overlap**. Two domains (cscw, jmir) actually
improved. Note OneIE was never re-measured under this stricter condition and
would very likely drop too, so 44.93 vs 41.61 understates the gap. Full
decomposition in `AUDIT.md` §D.

## 7. Honest scope notes

- **Segmentation is not modelled.** We consume the provided gold windows, exactly
  as OneIE, DEGREE and EEQA do. The paper's segmentation numbers are a separate
  zero-shot LLM experiment and are not comparable to a tuned pipeline. We report
  no segmentation number rather than an incomparable one.
- **The segment ordinal is deliberately unused.** The window id encodes the
  segment's position within its abstract. A majority-class rule on that ordinal
  alone predicts the window event type at **98.73 %** on dev, versus our model's
  **89.24 %** from text. Oracle typing is worth **+4.72 Arg-C / +5.91 Arg-I**, so
  most of that headroom is real and a deployed pipeline would legitimately have
  the feature. It is excluded from the headline system because the paper's
  baselines did not use it, and reported only as measured headroom.
- **Dev/test hygiene.** Architecture, checkpoint-selection rule, and all three
  decoding parameters were fixed on dev before a single test evaluation. No test
  statistic informed any decision. Every single contact with test data during
  development is enumerated in `AUDIT.md` §F, including one that changed nothing
  but is disclosed rather than omitted.
- **The benchmark's own split is window-level, not document-level**, contrary to
  the paper's description. Quantified and probed in §6.4 and `AUDIT.md` §C–D. The only prior contact with test was the
  gold-label oracle test that verifies the prediction exporter, which produces no
  model-selection signal.
- **Metric integrity.** The official evaluator is imported and its own functions
  are called; no metric is reimplemented, no threshold in it is altered. Verified
  by an oracle test that scores exactly 100.00 on every metric.

## 8. Remaining weaknesses

Ranked by how much they cost, from the frozen test analysis:

1. **Short spans (the dominant loss).** Recall at IoU>0.5 is **14.3 %** for gold
   spans of 1–4 tokens and 34.8 % for 5–9, against ~60 % for 10–34. This is
   self-inflicted: the frozen `min_len = 3` deletes short spans outright and
   τ = 0.90 suppresses the rest. It is the trade the dev tuner chose because
   precision was worth more, but a *length-conditioned* threshold (low τ for long
   spans, high τ for short ones) should recover much of it without the precision cost.
2. **Digital Humanities.** 28.6–30.8 Arg-C IoU against 72–75 for computational
   biology, stable across seeds. We reproduce the paper's humanities gap and do
   **not** close it. Neither domain adapters nor per-domain thresholds were tried.
3. **Rare roles are at zero.** Analysis (10 test instances), Contradictions (1),
   Ethical (1) all score 0.00. H6 (class-balanced loss) was deliberately not run
   because these are <2 % of gold, but the method has genuinely learned nothing
   about them.
4. **Conclusions/Implications** is the weakest event type (36.67), and the
   Implications role is weak (29.63) — the two are related.
5. **Window typing still costs 4.7 %** of gold arguments at 91.41 % accuracy.
6. **Exact-match remains low** (Arg-C EM 33.43 vs IoU 50.48). The method gets
   spans approximately right; it does not get them exactly right. That is
   adequate for this benchmark's headline metric and would not be for a stricter one.

## 9. Next experiments

1. **Length-conditioned decoding threshold** — highest expected value, near-zero
   cost, targets the single largest error bucket directly.
2. **Backbone sweep (H2 proper)** — ModernBERT-large is already downloaded;
   never run because three architectural hypotheses had already failed on
   DeBERTa-v3-large and a backbone swap was the lowest-information option left.
3. **Domain-aware treatment for DH** — per-domain thresholds first (cheap), then
   adapters if the gap is representational rather than calibration.
4. **Recover the event-type headroom legitimately** — the segment ordinal gives
   98.73 % type accuracy versus 91.41 % from text, and oracle typing is worth
   +4.72 Arg-C on dev. Defensible in a deployed pipeline; excluded here only for
   comparability with the paper's baselines.
5. **Ensembling that is compatible with calibration** — e.g. voting over decoded
   spans, or per-model calibration before averaging, rather than averaging raw
   posteriors.
6. **Revisit span enumeration** only if the model gets within ~1 F1 of the 99.70
   BIO ceiling; it is not the binding constraint today.

## 10. Reproducing

```bash
cd method/SciEvent-Next
pip install -r requirements.txt
bash scripts/reproduce.sh
```

Runtime: ~35 minutes total on one RTX 5880 Ada (3 × 10 min training plus
evaluation). Per-run records in `artifacts/EXPERIMENTS.md`, per-run logs in
`artifacts/runs/<run>/log.json`, frozen decoding rule in
`artifacts/decoding_rules.json`, test predictions in `artifacts/preds/`.
