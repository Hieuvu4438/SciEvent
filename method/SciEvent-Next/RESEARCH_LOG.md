# RESEARCH_LOG.md — SciEvent-Next

Hardware for every run: NVIDIA RTX 5880 Ada Generation (49 GB), CUDA 12.8,
torch 2.11.0+cu128, transformers 5.5.4, Python 3.13.5.
Data: `third_party/SciEvent/SciEvent_data/ONEIE/all_splits` — 1278 train /
158 dev / 163 test windows. All development on train+dev; test frozen.

Targets (paper, verified): **Arg-C IoU 41.61**, Arg-I IoU 53.57, trigger ROUGE-L 75.08.

---

## 2026-09-22 — Stage 0: benchmark contract and data geometry

Read the official evaluator (`baselines/ONEIE/EM_overlap_eval.py`) rather than
trusting the paper's prose. Three facts determine the whole design:

1. Arg-I / Arg-C are **trigger-insensitive** but **event-type sensitive**.
2. `{Agent, PrimaryObject, SecondaryObject}` are **excluded** from Arg-I / Arg-C;
   they are only used to build the ROUGE-L ⟨Agent, trigger, Primary, Secondary⟩
   tuple.
3. IoU matching is `intersection/union > 0.5`, greedy one-to-one per window.

Measured data geometry (train, 1278 windows): exactly one event per window;
mean 62 tokens/window; scored argument spans of mean 10–15 tokens; 97.6 % of
windows have **no overlap** among scored roles; modal gap between consecutive
spans is **0 tokens**. → the task is clause-level span segmentation, not mention
extraction. Full table in `LITERATURE.md` §0.

Role support (train): Context 1053, Method 968, Results 911, Challenge 406,
Purpose 254, Implications 235, Analysis 67, Contradictions 2, Ethical 1.

## 2026-09-22 — Stage 1: correctness tests

`tests/test_contract.py`:

| test | result |
|---|---|
| splits load with expected sizes | 1278 / 158 / 163 ✓ |
| BIO round-trip recovers gold role spans | 3857/3897 = **98.97 %** |
| **oracle**: gold → our exporter → official evaluator (dev) | **100.00** on Arg-C IoU, Arg-I IoU, Arg-C EM, ROUGE-L |
| **oracle** (test) | **100.00** |
| BIO-formulation ceiling (dev) | **99.70** Arg-C IoU |

The oracle test is the important one: it proves the prediction export is exact, so
every later number is trustworthy. The BIO ceiling of 99.70 confirms sequence
labelling costs ~0.3 F1 — the formulation is not the bottleneck.

## 2026-09-22 — Infrastructure defect found and fixed

First smoke run produced `loss nan` from epoch 1 with all metrics at 0.
Diagnosis: gradients were finite (grad-norm 11.02) but **all 390 parameter
tensors were NaN after a single `opt.step()` at lr 6.25e-07**. Cause:
`transformers` 5.5.4 honours the dtype stored in the checkpoint, and
`microsoft/deberta-v3-large` ships **fp16** weights, so AdamW was updating fp16
master weights. Fixed by loading the encoder with `dtype=torch.float32`
(mixed precision from autocast only). Recorded as D10.

## 2026-09-22 — Stage 2: smoke experiment (H1)

5 epochs, DeBERTa-v3-large, seed 42. Dev Arg-C IoU rose 14.20 → 33.20 and was
still climbing; ~19 s/epoch. Training path validated; promoted to a full run.

## 2026-09-22 — Stage 3: H1 full dev run

`configs/h1_deberta.json` — DeBERTa-v3-large, 40 epochs, bs 8, lr 1e-5/1e-4,
seed 42, checkpoint selected on dev Arg-C IoU F1.

*(results appended below as the run completes)*

### H1 full run — dev results (seed 42, DeBERTa-v3-large, 40 epochs, ~19 s/epoch)

Raw argmax decoding peaked at **Arg-C IoU 38.88** (P 31.8 / R 50.1), Arg-I IoU
47.46, trigger ROUGE-L 79.06, window-type accuracy 89.2 %.

**Diagnosis (`diagnose.py`), 497 gold scored arguments on dev:**

| bucket | count | share of gold |
|---|---|---|
| correct | 250 | 50.3 % |
| missed | 86 | 17.3 % |
| **killed by wrong window type** | 56 | 11.3 % |
| role confusion (span right, role wrong) | 53 | 10.7 % |
| boundary (role right, IoU ≤ 0.5) | 52 | 10.5 % |
| **spurious** | **388** | — |

Recall by gold span length: 1–4 tok **34.1**, 5–9 **41.3**, 10–19 **56.0**,
20–34 **64.3**, 35+ 60.0. Length bias on matched spans ≈ 0. So boundaries are
unbiased and long spans are handled well; short spans and over-prediction are the
problems.

The decisive number is **784 predicted spans against 497 gold**. Precision was
uniformly ~30 across every role while recall was 50–65 — a calibration failure,
not a representation failure.

### H5 — metric-aware decoding calibration (tuned on dev only)

Three parameters: span confidence τ (mean token posterior), minimum span length,
merge gap. Grid-searched on dev.

| decoding | pred spans | Arg-C IoU P/R/F1 | Arg-I IoU F1 | Arg-C EM F1 |
|---|---|---|---|---|
| argmax | 784 | 31.76 / 50.10 / **38.88** | 47.46 | 24.51 |
| **τ=0.7, min_len=3, merge_gap=0** | **519** | 46.05 / 48.09 / **47.05** | **56.89** | 28.94 |
| + per-role τ (9 extra params) | — | — / — / 47.72 | — | — |

**+8.17 Arg-C IoU from calibration alone.** Precision moved 31.8 → 46.1 for a
recall cost of only 2.0. The per-role refinement adds just +0.67 for nine extra
dev-fitted parameters on 158 windows, so the **global 3-parameter rule is
preferred** as the frozen rule; per-role thresholds are rejected as dev overfitting.

Interestingly `merge_gap=0` was selected: fragmentation is *not* the dominant
error, low-confidence spurious spans are. H4 (CRF) is therefore less likely to
help than initially predicted — tested next anyway.

**Status vs targets (dev, single seed):** Arg-C IoU 47.05 (+5.44),
Arg-I IoU 56.89 (+3.32), trigger ROUGE-L 79.06 (+3.98). All three exceeded.

**Per-domain Arg-C IoU (argmax decoding):** ACL 47.79, cscw 44.07, bioinfo 43.90,
jmir 36.81, **dh 29.39** — the humanities gap the paper reports is reproduced.

**Per-role Arg-C IoU F1 (argmax):** Challenge 52.63, Results 51.01,
Implications 40.00, Method 37.37, Context 30.46, Purpose 29.73, Analysis 12.90,
Ethical 0.00. Context is both the most frequent role and among the weakest.

### Event-type headroom (oracle diagnostic, H3)

| window type source | Arg-C IoU | Arg-I IoU |
|---|---|---|
| model prediction (89.24 % accurate) | 47.05 | 56.89 |
| **oracle gold type** | **51.77** | **62.80** |

Imperfect window typing costs **4.72 Arg-C / 5.91 Arg-I**. For transparency: a
majority-class rule on the segment's ordinal position within its abstract — a
feature any deployed pipeline has, encoded in the window id — reaches **98.73 %**
type accuracy on dev. It is deliberately **not** used by the headline system,
because the paper's baselines did not use it; it is reported as measured headroom
only. H3's explicit type head + conditioning is kept.

### Architecture ablations (all re-scored with identical tuned decoding on dev)

| run | recipe | Arg-C IoU | Arg-I IoU | Arg-C EM | ROUGE-L | tuned τ |
|---|---|---|---|---|---|---|
| `h1_deberta_s42` | lr 1e-5, no LLRD, 40 ep | **47.05** | **56.89** | 28.94 | 79.06 | 0.70 |
| `h3_llrd_s42` | lr 2e-5, LLRD 0.9, 30 ep | 46.58 | 54.60 | 27.40 | 78.34 | 0.65 |

**H2/LLRD verdict: rejected.** Layer-wise LR decay converges faster (Arg-C 44 by
epoch 11 vs epoch 20) but lands 0.47 lower — inside seed noise, no benefit.
The simpler recipe is kept.

### H5 revisited — the threshold curve is a plateau, not a spike

Fine sweep of τ on dev for `h1_deberta_s42` (min_len = 3):

| τ | Arg-C P | Arg-C R | **Arg-C F1** | Arg-I F1 | #pred (gold 497) |
|---|---|---|---|---|---|
| 0.50 | 43.22 | 48.09 | 45.52 | 55.43 | 553 |
| 0.70 | 46.05 | 48.09 | 47.05 | **56.89** | 519 |
| 0.80 | 48.34 | 46.88 | 47.60 | 56.79 | 482 |
| 0.85 | 49.89 | 45.67 | 47.69 | 55.88 | 455 |
| **0.90** | 53.60 | 43.46 | **48.00** | 55.56 | 403 |
| 0.93 | 55.56 | 40.24 | 46.67 | 53.21 | 360 |
| 0.95 | 59.55 | 37.63 | 46.12 | 51.29 | 314 |
| 0.99 | 69.54 | 24.35 | 36.07 | 38.75 | 174 |

The optimum is a **broad plateau across τ ∈ [0.7, 0.9]** spanning ~1 F1, with a
sharp collapse only beyond 0.93. That is the signature of a genuine calibration
correction, not of fitting dev noise. Arg-C peaks at τ=0.90 while Arg-I peaks
earlier (τ=0.70); the declared selection criterion is **dev Arg-C IoU F1**, the
paper's primary metric, and Arg-I stays above its 53.57 target throughout the
plateau. Grid extended so the optimum is interior.

`min_len=3` forfeits at most 7.6 % of dev gold spans (those of length ≤ 2) and is
worth ~1.5 precision points; the grid selects it over `min_len ∈ {1,2,4}`.

### H4 — CRF: falsified

All runs re-scored with the *same* widened decoding grid on dev (seed 42):

| run | recipe | argmax Arg-C | **tuned Arg-C IoU** | Arg-I IoU | Arg-C EM | ROUGE-L | τ* |
|---|---|---|---|---|---|---|---|
| `h1_deberta_s42` | plain (lr 1e-5, 40 ep) | 38.88 | **48.00** | 55.56 | 30.00 | 79.06 | 0.90 |
| `h3_llrd_s42` | + LLRD 0.9, lr 2e-5 | 36.85 | 46.58 | 54.60 | 27.40 | 78.34 | 0.65 |
| `h4_crf_s42` | + LLRD + **CRF** | 35.82 | 46.52 | 54.50 | 30.79 | 79.40 | 0.80 |

Isolating each change against its own control:
- **LLRD: −1.42 → rejected.** Converges faster (Arg-C 44 by epoch 11 vs 20) but
  lands lower.
- **CRF: −0.06 vs its LLRD-matched control → falsified.** The hypothesis was that
  learned transitions would suppress fragmentation across the zero-token gaps
  between adjacent gold spans. They do not help, and this is consistent with the
  H5 finding that the tuned `merge_gap` is **0**: fragmentation was never the
  dominant error. Over-prediction of low-confidence spans was, and a confidence
  threshold addresses that directly and more cheaply than a CRF.

Both changes are dropped. The plain recipe is retained.

### H8 — span-level role re-classification: falsified

Trained with the auxiliary span-role head (seed 42, LLRD recipe as its control),
then evaluated with the span override on and off, identical decoding grid:

| configuration | Arg-C IoU | Arg-I IoU | Arg-C EM |
|---|---|---|---|
| `h3_llrd_s42` (control, no span head) | 46.58 | 54.60 | 27.40 |
| `h8_span_s42`, override **off** | 46.56 | 53.42 | — |
| `h8_span_s42`, override **on** | **46.80** | 53.42 | 27.63 |

**Gain from the span-level role head: +0.24** against its own control — below the
0.5 falsification threshold, so **H8 is rejected**. Worth recording *why* the
prediction failed: the diagnosis said 53/497 gold arguments were "span right,
role wrong", so a span-level view looked like the obvious fix. But once the
confidence threshold removes the low-confidence spans, most of those role
confusions are removed along with them — they were largely the *same* errors,
counted twice. The auxiliary loss also cost 1.2 Arg-I, so the head is dropped.

### Ablation summary (seed 42, identical tuned decoding on dev)

| run | Arg-C IoU | Arg-I IoU | Arg-C EM | ROUGE-L | verdict |
|---|---|---|---|---|---|
| **`h1_deberta_s42` (plain)** | **48.00** | **55.56** | 30.00 | 79.06 | **kept** |
| `h8_span_s42` (+LLRD +span-role) | 46.80 | 53.42 | 27.63 | 78.54 | H8 rejected |
| `h3_llrd_s42` (+LLRD) | 46.58 | 54.60 | 27.40 | 78.34 | LLRD rejected |
| `h4_crf_s42` (+LLRD +CRF) | 46.52 | 54.50 | 30.79 | 79.40 | H4 falsified |

Three genuinely different architectural hypotheses (structured decoding,
layer-wise adaptation, span-level role modelling) were each tested and each
failed to beat the plain multi-task tagger. The remaining lever is H7
(multi-seed posterior ensembling), which is now run on the frozen recipe.

## Stage 4: frozen final method

`configs/final.json` — DeBERTa-v3-large, 30 epochs, bs 8, lr 1e-5 / 1e-4, no
LLRD, no CRF, no span-role head; checkpoint selected on dev Arg-C IoU F1 under
a swept confidence threshold. Seeds 42 / 13 / 101, posteriors averaged (H7),
decoding rules tuned on dev, then a single frozen test evaluation.

### H7 — posterior ensembling: falsified

3-seed posterior average, tuned on dev with the common grid:

| system | argmax Arg-C | tuned Arg-C IoU | Arg-I IoU |
|---|---|---|---|
| seed 42 alone | 36.69 | 47.66 | 55.50 |
| seed 13 alone | 38.52 | **48.10** | 57.31 |
| seed 101 alone | 37.32 | 46.99 | 54.64 |
| **3-seed posterior average** | **39.35** | 47.10 | 54.11 |

Ensembling clearly helps **argmax** decoding (39.35 vs 36.7–38.5) but *hurts*
**calibrated** decoding (47.10, below the seed mean of 47.58 and well below the
best seed). Averaging compresses the posterior distribution that the confidence
threshold operates on, so the very signal H5 exploits is blunted. **H7 rejected.**

The shipped method is therefore a **single model**, and the honest report is the
mean ± std over three seeds.

### Freezing the decoding rule (seed-agnostic)

To avoid fitting the threshold to whichever seed looks best on dev, the rule was
chosen to maximise the **mean dev Arg-C IoU across all three seeds**
(`scripts/freeze_rules.py`). The τ = 0.88–0.92 region is a low-variance plateau:

| τ | min_len | s42 | s13 | s101 | mean | std |
|---|---|---|---|---|---|---|
| 0.85 | 3 | 46.62 | 47.60 | 45.95 | 46.73 | 0.83 |
| 0.88 | 3 | 46.91 | 47.05 | 46.58 | 46.85 | 0.24 |
| **0.90** | **3** | 47.40 | 47.15 | 46.87 | **47.14** | **0.26** |
| 0.92 | 3 | 47.34 | 46.94 | 45.84 | 46.71 | 0.78 |

**FROZEN: τ = 0.90, min_len = 3, merge_gap = 0.** Dev Arg-C IoU **47.14 ± 0.26**.
Nothing further was tuned.

## Stage 6 — single frozen evaluation on TEST

Checkpoints, architecture and decoding rule all fixed above. Test touched once.

| metric | mean ± std | min | max | paper best | Δ (mean) |
|---|---|---|---|---|---|
| **Arg-C IoU F1** | **50.48 ± 1.15** | 49.22 | 51.48 | 41.61 | **+8.87** |
| **Arg-I IoU F1** | **57.95 ± 2.46** | 55.24 | 60.06 | 53.57 | **+4.38** |
| **Trigger ROUGE-L F1** | **76.93 ± 0.80** | 76.02 | 77.52 | 75.08 | **+1.85** |
| Arg-C EM F1 | 33.43 ± 0.79 | 32.81 | 34.32 | — | — |
| Arg-I EM F1 | 36.56 ± 0.95 | 35.72 | 37.59 | — | — |
| Arg-C overlap F1 | 59.11 ± 1.22 | — | — | — | — |
| Arg-C SciREX F1 | 55.03 ± 1.46 | — | — | — | — |

**Every individual seed beats every target** (worst-seed margins: Arg-C +7.61,
Arg-I +1.67, ROUGE-L +0.94). Test is *above* dev (50.48 vs 47.14), so the
dev-tuned threshold did not overfit dev.

### Test error analysis (seed 13)

| bucket | count | share of 533 gold |
|---|---|---|
| correct | 246 | 46.2 % |
| **missed** | **177** | **33.2 %** |
| boundary | 44 | 8.3 % |
| role confusion | 41 | 7.7 % |
| wrong window type | 25 | 4.7 % |
| spurious | 102 | — |

Window-type accuracy on test is **91.41 %** (better than dev's 89.24 %), so type
errors now cost only 4.7 % of gold.

Recall by gold span length is the clearest remaining weakness:

| span length | n | recall @ IoU>0.5 |
|---|---|---|
| 1–4 | 77 | **14.3** |
| 5–9 | 141 | 34.8 |
| 10–19 | 202 | 58.4 |
| 20–34 | 106 | 61.3 |
| 35+ | 7 | 42.9 |

The frozen `min_len = 3` deletes short spans outright and τ=0.90 suppresses the
rest, which is precisely the precision/recall trade the tuner chose. Length bias
on matched spans is ~0, so boundaries themselves are unbiased.
