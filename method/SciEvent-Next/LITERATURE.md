# LITERATURE.md — SciEvent-Next

Scope: what the relevant literature says, and — more importantly — which of it
actually transfers to *this* benchmark once its data geometry is measured.

## 0. The measurement that reframes the literature

Before reading any EAE paper, we measured the SciEvent training split
(`third_party/SciEvent/SciEvent_data/ONEIE/all_splits/train.oneie.json`, 1278 windows):

| property | value |
|---|---|
| events per window | **exactly 1**, in all 1278/158/163 windows |
| window length | mean 62 whitespace tokens, p95 123, max 260 |
| scored arguments per window | mean 3.05 (0–24) |
| **scored argument span length** | mean 10–15 tokens (Context 10.2, Method 13.2, Results 14.5, Challenge 14.4, Analysis 15.5) |
| windows with overlap *within* scored roles | **2.4 %** |
| windows with overlap *within* Agent/PrimaryObject/SecondaryObject | **0.2 %** |
| windows with *cross-group* overlap (AAO vs scored) | 5.9 % |
| most common gap between consecutive scored spans | **0 tokens** (942), then 1 (658) |
| token coverage of a window by scored spans | 56 % |

**Conclusion.** SciEvent argument extraction is *not* the classic EAE problem of
picking short entity mentions out of a sentence. It is **clause-level span
segmentation with rhetorical/semantic role labelling** over a short passage that
contains exactly one event. The spans are long, contiguous, adjacent and
essentially non-overlapping.

Almost every design decision below follows from that one table.

## 1. Papers reviewed, and what actually transfers

### Directly formative

**Are Triggers Needed for Document-Level Event Extraction?** (Cornell, TACL 2025,
[arXiv 2411.08708](https://arxiv.org/pdf/2411.08708), [TACL](https://aclanthology.org/2025.tacl-1.71/))
- *Finding*: whether systems benefit from explicitly extracting triggers depends
  on the number of events per document and on whether a natural-language schema
  is available; with few events per document, trigger extraction is largely
  redundant. Random triggers still help *prompt-based ICL*, but not trained models.
- *Transfer*: **decisive**. SciEvent has exactly one event per window, and the
  official Arg-I/Arg-C metrics are explicitly *trigger-insensitive*. So the
  trigger is not on the critical path for the headline metric at all. It is only
  needed for the separate ROUGE-L tuple score. → H1 decouples them.

**Joint Span Segmentation and Rhetorical Role Labeling** (Modi et al.,
[arXiv 2302.06448](https://arxiv.org/abs/2302.06448))
- *Finding*: for legal documents, the right formulation is spans of consecutive
  units sharing a rhetorical role, learned jointly with semi-Markov CRF.
- *Transfer*: **high**. This is structurally the same problem as SciEvent
  arguments (contiguous clause spans carrying discourse-semantic roles). It
  justifies sequence labelling over span enumeration, and flags semi-Markov / CRF
  transition modelling as the natural refinement (→ H4).

**PAIE / span-selection EAE** and the wider boundary-modelling line
- *Finding*: span-selection with strong encoders (DeBERTa-v3-large is the
  standard backbone) beats generation for boundary fidelity.
- *Transfer*: backbone choice and the start/end-vs-BIO decision (→ H2 tests
  encoders; BIO chosen because spans are contiguous and non-overlapping).

**DEGREE** (generative, templated) — *measured on SciEvent by the authors*:
Arg-I IoU **67.79 P / 19.13 R**, Arg-C IoU 48.99 P / 13.83 R.
- *Transfer as a negative result*: the recall collapse is the signature of a
  generative model being asked to copy 10–15-token spans verbatim. This is strong
  evidence **against** starting from constrained generation here, and is why we
  do not begin with a LoRA-tuned LLM.

**OneIE** (joint graph decoding) — the reference to beat: Arg-I IoU **53.57**,
Arg-C IoU **41.61**. Its balanced but low P/R (39.69/43.71 on Arg-C) is the
profile of an entity-mention-oriented model applied to clause-sized spans.

**EEQA** (QA-style) — Arg-I IoU 32.91 / Arg-C IoU 26.51, and trigger ROUGE-L with
81.93 P but 34.57 R. QA extraction of one span per (question, passage) is a poor
fit for a role that occurs multiple times per window (Context repeats in 252/1278
training windows).

### Reviewed, transferable ideas noted, not adopted first

| work | idea | why not first |
|---|---|---|
| **GEMS** (Findings ACL 2025, [anthology](https://aclanthology.org/2025.findings-acl.1353/)) | multi-perspective prompts + ontology steering + voting over argument orders | generation-based; DEGREE's 13.8 Arg-C recall on this benchmark is the counter-evidence. Its *voting/ensembling* idea is retained as H7. |
| **REGen** (Findings EMNLP 2025) | relaxed + LLM-based matching for generative EAE | evaluation-side; SciEvent already uses relaxed (IoU) matching, and we must not alter the evaluator. |
| **DEGAP** (dual event-guided adaptive prefixes) | prefix tuning with retrieved event context | orthogonal; 1278 training windows make prefix retrieval thin. |
| Retrieve-and-Sample / doc-level EAE | hybrid retrieval augmentation | windows are 62 tokens — there is no document-level long-range problem to solve. |
| Scientific discourse tagging (arXiv 1909.04758) | discourse-role tagging of scientific text | confirms the sequence-labelling framing; its label set is coarser than SciEvent's. |
| MoE / domain adapters | per-domain specialisation | 5 domains × ~250 train windows each; too thin. Held as H6 only if domain analysis shows a real gap. |

### Checked: is there a newer directly comparable result?

A search of post-SciEvent literature found no paper reporting Arg-C/Arg-I on the
Dong et al. SciEvent benchmark. Several similarly named datasets exist and were
excluded. **The targets therefore remain the paper's own: Arg-C IoU 41.61,
Arg-I IoU 53.57, trigger ROUGE-L 75.08.**

## 2. Ranked research-opportunity map

Ordered by (expected score gain × probability) ÷ (compute + code cost).

1. **Reformulate argument extraction as word-level span segmentation.** The whole
   baseline field treated this as mention-style EAE. The data says it is
   segmentation. Highest expected gain, lowest cost. → **H1**
2. **Exploit that the event type is a window-level 4-way label.** The metric is
   event-type-sensitive and trigger-insensitive, so window type accuracy acts as
   a multiplier on *every* argument. Model it explicitly and condition the role
   tagger on it. → **H1 / H3**
3. **Backbone strength on long-span boundaries.** DeBERTa-v3-large's
   disentangled attention is the known best token-labelling backbone. → **H2**
4. **Structured decoding over BIO transitions** (CRF / semi-Markov), which the
   legal rhetorical-role work shows matters when spans are long and adjacent. → **H4**
5. **Metric-aware decoding.** Matching is IoU>0.5 one-to-one; span *count*
   calibration and merging of over-fragmented spans trade P against R directly. → **H5**
6. **Rare-role treatment** (Purpose 254, Implications 235, Analysis 67,
   Contradictions 2, Ethical 1) — class-balanced loss. → **H6**
7. **Seed/checkpoint ensembling** over the tagger's per-token posteriors. → **H7**

## 3. Why not an LLM + LoRA first

It is a legitimate candidate and is kept as a fallback, but on this benchmark:
generation must copy 10–15-token spans exactly enough to clear IoU>0.5, the
training set is 1278 windows, the reported generative baseline loses on recall by
a factor of three, and a 0.4B-parameter encoder trains in minutes on the
available GPU while a 7B LoRA run costs hours per hypothesis. The information
gain per GPU-hour is far higher on the discriminative side, so the research loop
starts there and escalates only if it plateaus below target.

Sources: [SciEvent (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.871/),
[Are Triggers Needed? (TACL 2025)](https://aclanthology.org/2025.tacl-1.71/),
[GEMS (Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.1353/),
[Joint Span Segmentation and Rhetorical Role Labeling](https://arxiv.org/abs/2302.06448),
[REGen (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.649/),
[Scientific Discourse Tagging](https://arxiv.org/pdf/1909.04758).
