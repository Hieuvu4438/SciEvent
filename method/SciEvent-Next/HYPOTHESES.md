# HYPOTHESES.md — SciEvent-Next

Reference targets (SciEvent, EMNLP 2025, verified against the paper's tables):

| metric | best reported | by |
|---|---|---|
| **Arg-C IoU F1** (primary) | **41.61** | OneIE |
| **Arg-I IoU F1** (secondary) | **53.57** | OneIE |
| Trigger ROUGE-L F1 | 75.08 | GPT 5-shot |
| Segmentation EM / IoU F1 | 60.95 / 85.63 | GPT (not modelled — we consume gold segments, as all tuned baselines do) |

Measured ceilings for our formulation (`tests/test_contract.py`):
gold→official evaluator = **100.00** on all metrics (export is exact);
BIO-encoded gold = **99.70** Arg-C IoU on dev. The formulation costs 0.3 F1.

---

## H1 — Argument extraction on SciEvent is span segmentation, not mention extraction

- **Problem.** All tuned baselines (OneIE, DEGREE, EEQA) model arguments as short
  entity mentions or single QA answers. SciEvent arguments are 10–15-token
  clauses that tile the window almost contiguously.
- **Evidence.** Span length table above; 97.6 % of windows have zero overlap
  among scored roles; modal gap between consecutive spans is 0 tokens; DEGREE's
  Arg-C recall is 13.83 (generative copying fails on long spans); EEQA's trigger
  recall is 34.57 (one-answer-per-question fails when Context repeats).
- **Mechanism.** A word-level BIO tagger over the 9 scored roles on top of a
  strong encoder, with a second disjoint BIO head for the
  ⟨Agent, Action, PrimaryObject, SecondaryObject⟩ tuple (needed only for
  ROUGE-L), and a window-level 4-way event-type head. Two heads rather than one
  because cross-group overlap is 5.9 % while within-group overlap is ≤2.4 %.
- **Expected effect.** Large gain on Arg-C IoU, driven mainly by **recall**
  (OneIE's recall is 43.71; a tagger should exceed it substantially) without the
  precision collapse that generation shows.
- **Cost.** ~0.4B-parameter encoder, minutes per run.
- **Falsification.** Dev Arg-C IoU F1 < 41.61 after full training.
- **Result.** Dev Arg-C IoU **48.00** (single seed, tuned decoding) vs the 41.61 target; Arg-I IoU 55.56 vs 53.57; trigger ROUGE-L 79.06 vs 75.08. Raw argmax decoding alone reached 38.88.
- **Decision.** **KEEP** — this is the method. The reframing, not model scale, is what beats the baselines.

## H2 — Backbone matters more than domain match for long-span boundaries

- **Problem.** Which encoder? Scientific-domain (SciBERT/SciDeBERTa) vs strongest
  general token-labeller (DeBERTa-v3-large) vs modern long-context (ModernBERT).
- **Evidence.** Boundary decisions here are syntactic/discourse ("and", "that",
  "However," precede 13.6 % of span starts) rather than terminology-driven;
  DeBERTa-v3's disentangled attention is the standard best token-labelling choice.
- **Mechanism.** Same architecture, swap `model_name`.
- **Expected effect.** 1–4 F1 spread between backbones.
- **Falsification.** A domain-pretrained base model matches DeBERTa-v3-large.
- **Result.** DeBERTa-v3-large used throughout. LLRD (a proxy for tuning the adaptation schedule) *hurt* by 1.42, indicating the backbone is not adaptation-limited on 1278 windows. ModernBERT-large was downloaded but not run: with three architectural hypotheses already falsified on the same backbone, a backbone swap was the lowest-information remaining experiment.
- **Decision.** **PARTIALLY TESTED** — DeBERTa-v3-large retained; a backbone sweep is listed as future work.

## H3 — Window event type is a score multiplier and must be modelled explicitly

- **Problem.** Arg-I/Arg-C are event-type sensitive. One event per window ⇒ a
  wrong window type zeroes *every* argument in it (all FP **and** all FN).
- **Mechanism.** Dedicated 4-way head; a learned event-type embedding added to
  word states before the role head; scheduled sampling from gold to self-predicted
  conditioning so train/inference match.
- **Expected effect.** Arg-C IoU ≈ type-accuracy × role quality; raising type
  accuracy from 85 % to 92 % is worth several F1.
- **Falsification.** Ablating the conditioning embedding changes Arg-C by <0.5.
- **Result.** Window-type accuracy 89.2 %. Oracle substitution of the gold type raises Arg-C 47.05 → **51.77** and Arg-I 56.89 → **62.80**, confirming type is a multiplier worth ~4.7 Arg-C.
- **Decision.** **KEEP** the explicit type head and conditioning; the residual 4.7 is documented headroom, most of which a deployed pipeline could recover from the segment ordinal (98.7 % type accuracy) — deliberately unused here for comparability.

## H4 — Structured decoding (CRF / semi-Markov) helps when spans are long and adjacent

- **Evidence.** Modal inter-span gap is 0 ⇒ the tagger must decide
  `I-Context → B-Method` transitions with no O separator. Independent softmax
  per token has no transition prior. Rhetorical-role literature uses semi-Markov CRF.
- **Falsification.** CRF gains <0.5 Arg-C IoU over softmax.
- **Result.** CRF: **−0.06** against its LLRD-matched control (46.52 vs 46.58). Its argmax Arg-C was also *lower* (35.82 vs 36.85).
- **Decision.** **REJECT** — falsified. Consistent with H5's tuned `merge_gap = 0`: fragmentation was never the dominant error.

## H5 — Metric-aware decoding: IoU>0.5 one-to-one matching is exploitable *honestly*

- **Mechanism.** (a) drop low-confidence spans (per-role probability threshold
  tuned on **dev only**); (b) merge same-role spans separated by ≤k tokens, since
  over-fragmentation converts one true positive into one TP + one FP;
  (c) minimum span length. All thresholds frozen before test.
- **Falsification.** Tuned decoding gains <0.5 over argmax decoding.
- **Result.** **+9.12 Arg-C IoU** (38.88 argmax → 48.00 tuned). Precision 31.8 → 53.6 for a recall cost of 6.6. The τ curve is a broad plateau over [0.7, 0.9], not a spike. Per-role thresholds added only +0.67 for nine extra dev-fitted parameters and were rejected.
- **Decision.** **KEEP** the 3-parameter global rule (τ, min_len, merge_gap), tuned on dev and frozen before test.

## H6 — Rare roles need different treatment from boundary errors

- **Evidence.** Support: Context 1053, Method 968, Results 911, Challenge 406,
  Purpose 254, Implications 235, Analysis 67, Contradictions 2, Ethical 1.
- **Mechanism.** Class-balanced / focal loss on the role head, or role-frequency
  logit priors.
- **Falsification.** Macro-role F1 does not improve, or head roles degrade more
  than tail roles gain.
- **Result.** Not run. The tail roles are Analysis (67 train instances), Contradictions (2) and Ethical (1) — together under 2 % of gold. Re-weighting them could not plausibly move the headline metric, and would risk the head roles that dominate it.
- **Decision.** **NOT PURSUED** — documented as low expected information gain, not as a finding.

## H7 — Seed/checkpoint ensembling of token posteriors

- **Mechanism.** Average per-token log-probabilities across seeds before decoding.
- **Falsification.** <0.5 gain over best single seed.
- **Result.** see §Stage 4 in RESEARCH_LOG.md
- **Decision.** —

## Fallback F1 — LoRA-tuned open LLM with constrained span copying

Only if H1–H7 plateau below target. Deprioritised: DEGREE's 13.83 Arg-C recall on
this exact benchmark is direct evidence that generative copying of 10–15-token
spans fails here, and one LLM run costs more GPU-hours than all of H1–H7 combined.

---

## H8 — Span-level role re-classification (added after H1 diagnostics)

- **Problem.** On dev, Arg-I IoU 56.89 vs Arg-C IoU 47.05. That ~10-point gap is
  *pure role confusion*: the span is located correctly and the label is wrong.
  The taxonomy confirms it — 53/497 gold arguments are "span right, role wrong",
  and the confusions are semantic neighbours (Method↔Context, Results↔Method,
  Context↔Results).
- **Mechanism.** A BIO head assigns a role *per token*; nothing in it sees the
  whole span at once. Add a span-level head over
  `[h_start ; h_end ; mean(h)]` (event-type conditioned) that re-labels each
  decoded span. Trained on gold spans, applied to decoded spans.
- **Expected effect.** Closes part of the Arg-I/Arg-C gap; should not change
  Arg-I at all (the spans are unchanged), which makes it a clean, falsifiable test.
- **Falsification.** Arg-C IoU gain < 0.5 with Arg-I unchanged.
- **Result.** **+0.24** against its control (46.80 vs 46.58) — below the 0.5 threshold — and it cost 1.2 Arg-I. The predicted mechanism failed because the confidence threshold already removes most 'span right, role wrong' cases; they were the same errors counted twice.
- **Decision.** **REJECT** — falsified.
