# New experiments for the NAACL submission

Findings from the round of experiments run to close the reviewer attack surfaces
identified in the venue assessment. Numbers here supersede any earlier claim they
contradict.

---

## 1. Bootstrap confidence intervals — **changes what we may claim**

2000 bootstrap resamples over the 163 test windows (the unit of annotation),
re-scored with the benchmark's own matching functions, averaging the three seeds
inside each replicate.

| metric | point | 95 % CI | published baseline | CI lower − baseline | resamples beating baseline |
|---|---|---|---|---|---|
| **Arg-C IoU** (primary) | 50.38 | **[45.49, 55.48]** | 41.61 | **+3.88** | **100.00 %** |
| Arg-I IoU | 57.88 | [53.40, 62.37] | 53.57 | −0.17 | 97.10 % |
| Trigger ROUGE-L | 76.96 | [72.68, 80.90] | 75.08 | −2.40 | 82.05 % |

**Consequence for the paper — the three metrics must be claimed differently:**

- **Arg-C IoU: a clear, statistically solid improvement.** Every resample beats
  the baseline and the interval's lower bound is still 3.88 above it. This is the
  headline and it survives scrutiny.
- **Arg-I IoU: an improvement, but marginal.** The interval includes the
  baseline by 0.17. Report as an improvement, state the 97.1 %, do not claim
  significance.
- **Trigger ROUGE-L: NOT a significant improvement.** Only 82 % of resamples beat
  the baseline. With 163 windows this metric is too noisy to support a claim.
  **Report as comparable, not as +1.85.**

A paired test against OneIE is impossible: the SciEvent paper reports aggregate
numbers only and per-window baseline predictions are not released. The interval
above is the strongest claim the available evidence supports.

**Validity of the bootstrap.** The resampling needs per-window match counts, which
the official scorer does not expose, so `scripts/bootstrap_ci.py` re-implements the
accumulation loop. That re-implementation is verified to reproduce the official
scorer **exactly** on the identity (non-resampled) index:

| seed | official Arg-C | bootstrap loop | diff | official Arg-I | bootstrap loop | diff |
|---|---|---|---|---|---|---|
| 42 | 49.221184 | 49.221184 | 0.00e+00 | 55.244029 | 55.244029 | 0.00e+00 |
| 13 | 51.481103 | 51.481103 | 0.00e+00 | 60.061287 | 60.061287 | 0.00e+00 |
| 101 | 50.749251 | 50.749251 | 0.00e+00 | 58.541459 | 58.541459 | 0.00e+00 |

Matching itself uses the benchmark's own `iou_overlap`; only the accumulation is
ours. Note the observed 3-seed mean is **50.48**; the 50.38 in the table above is
the mean over bootstrap replicates, which differs slightly by construction. Report
50.48 as the point estimate and [45.49, 55.48] as its interval.

## 2. Length-conditioned decoding threshold — **tested, rejected**

The shipped rule drops any span shorter than `min_len = 3` regardless of
confidence. We predicted this was self-inflicted and that a finite short-span
threshold would recover much of the 14.3 % recall on 1–4 token gold spans.

Rule family searched (same three parameters, so no added dev-fitted capacity):

```
tau_eff(len) = tau_short   if len <  short_len
             = tau_long    if len >= short_len
```

Grid: `tau_long ∈ {0.80…0.92}`, `tau_short ∈ {0.90…0.99}`, `short_len ∈ {3,4,5}`,
selected on the mean dev Arg-C IoU across the three seeds.

**Result: +0.04** (47.18 ± 0.21 vs the shipped 47.14 ± 0.26). The tuner selects
`tau_short = 0.99`, i.e. it chooses to keep discarding short spans.
**The shipped rule is unchanged, so no test re-evaluation was needed.**

### Why it fails — the mechanism (dev, seed 42)

Predicted spans from the ROLE head, bucketed by predicted length:

| predicted length | #correct | #wrong | precision | median conf (correct) | median conf (wrong) |
|---|---|---|---|---|---|
| **1–2** | **7** | **225** | **3.0 %** | 0.910 | 0.632 |
| 3–4 | 15 | 89 | 14.4 % | 0.963 | 0.819 |
| 5–9 | 55 | 130 | 29.7 % | 0.967 | 0.911 |
| 10+ | 179 | 95 | **65.3 %** | 0.997 | 0.972 |

And confidence carries **no usable signal** for short spans: among predicted
spans of length < 3 with confidence ≥ 0.95, **0 of 5 are correct**.

**Interpretation.** Short outputs from the ROLE head are not deliberate short
predictions — they are *fragments* of a clause-level segmentation. The head is
trained on 10–15-token targets, so when it emits a 1–2 token span it is failing,
not making a fine-grained decision. `min_len = 3` is therefore the correct rule,
not a limitation of the decoding, and the short-span weakness is a representation
property rather than a calibration artifact.

**This supersedes the earlier claim** that "a length-conditioned threshold should
recover much of it". It was tested and it does not.

## 3. Controlled geometry test — **refutes the claim as we had stated it**

The same windows contain two groups of target spans, extracted by the *same
model* in the *same forward pass*, matched with the benchmark's own
`iou_overlap`: the 9 scored roles (clause-sized) and Agent / Primary /
SecondaryObject (mention-sized). Everything else is held constant.

| target | group | mean length | gold | P | R | F1 |
|---|---|---|---|---|---|---|
| **Agent** | mention-sized | **2.4** | 163 | 79.26 | 82.82 | **81.00** |
| SecondaryObject | mention-sized | 5.3 | 11 | 0.00 | 0.00 | 0.00 |
| PrimaryObject | mention-sized | 5.9 | 153 | 53.33 | 63.18 | 57.81 |
| Purpose | clause-sized | 10.9 | 43 | 57.97 | 31.01 | 40.40 |
| **Context** | clause-sized | **11.0** | 142 | 37.61 | 30.99 | **33.97** |
| Method | clause-sized | 13.2 | 143 | 65.45 | 59.67 | 62.36 |
| Results | clause-sized | 13.3 | 116 | 63.49 | 65.23 | 64.32 |
| Implications | clause-sized | 14.0 | 19 | 65.75 | 31.58 | 41.93 |
| Challenge | clause-sized | 14.6 | 58 | 70.28 | 58.05 | 63.56 |
| Analysis | clause-sized | 18.5 | 10 | 0.00 | 0.00 | 0.00 |
| | | | | | | |
| **mention-sized (pooled)** | | | 327 | | 70.85 | **67.43** |
| **clause-sized (pooled)** | | | 531 | | 48.59 | **51.65** |

**The mention-sized group scores higher than the clause-sized group** (67.43 vs
51.65), and the shortest target of all — Agent, 2.4 tokens — is the best of any
(81.00), while Context at 11.0 tokens is among the worst (33.97).

**Consequences:**

1. **Span length does not drive difficulty.** Role semantics does. Context is the
   hardest role despite being clause-sized; Agent is the easiest despite being
   mention-sized.
2. **The earlier plan to use RAMS as a "negative control" is unsupported.** It
   rested on the prediction that this architecture fails on mention-sized
   targets. It does not. Running RAMS now would be an uncontrolled guess, not a
   test of a hypothesis, and is therefore dropped rather than reported.
3. **The two-head design is vindicated**, and this is the ablation-grade evidence
   for it: each head handles the geometry it was trained for.
4. **The paper's claim must be narrowed** to what the evidence supports: the
   *scored* arguments of SciEvent are clause-sized and near-contiguous, which is
   why a segmentation formulation beats mention-extraction machinery on *this*
   metric. Not the general claim that "task geometry determines model family".

## 4. Ablation suite rebuilt — the previous one was invalid

The earlier ablations differed from the full recipe by **four factors at once**
(lr 2e-5 vs 1e-5, dropout 0.15 vs 0.1, `w_type` 1.0 vs 0.5, LLRD 0.9 vs 1.0) and
were additionally tuned on different decoding grids. No row of that table was a
valid single-factor comparison.

Two core design claims had also never been ablated at all: event-type
conditioning (only an oracle diagnostic existed) and the two-head split.

The suite was rebuilt so each configuration differs from the full model by
**exactly one factor**, run on the **same three seeds**, and re-scored on the
**same decoding grid**:

| configuration | differs by |
|---|---|
| Full model | — |
| − event-type conditioning | `use_type_cond: False` |
| − two disjoint heads | `single_head: True` (13 merged span types, one BIO layer) |
| + linear-chain CRF | `use_crf: True` |
| + span-level role head | `use_span_role: True` |
| + layer-wise LR decay | `llrd: 0.9` |

### Results (dev, 3 seeds, identical decoding grid)

| configuration | dev Arg-C IoU F1 | Δ |
|---|---|---|
| **Full model (CARVE)** | **47.58 ± 0.55** | — |
| − event-type conditioning | 47.34 ± 1.61 | −0.24 |
| − two disjoint heads | 48.34 ± 1.60 | **+0.75** |
| + linear-chain CRF | 47.30 ± 0.50 | −0.28 |
| + span-level role head | 46.63 ± 1.59 | −0.96 |
| + layer-wise LR decay | 45.38 ± 1.18 | −2.20 |

Per-seed values for the two surprising rows:

| configuration | seed 42 | seed 13 | seed 101 | mean ± std |
|---|---|---|---|---|
| Full model | 47.40 | 48.10 | 46.87 | 47.46 ± 0.61 |
| − two disjoint heads | **50.10** | 46.97 | 47.85 | 48.31 ± 1.62 |
| − event-type conditioning | 48.07 | 48.76 | 45.41 | 47.41 ± 1.77 |

### **Two of our four design decisions do not survive ablation**

This is the most consequential result of the round and it changes §4 of the paper.

**1. Event-type conditioning is inert (−0.24, std 1.61).** We justified it with an
oracle diagnostic showing that substituting the *gold* window type is worth
+4.72 Arg-C. That diagnostic is correct, but it establishes only that **getting
the type right matters** — it does **not** establish that **conditioning the role
head on the predicted type helps**. Those are different claims and we conflated
them. The type head itself remains necessary (the metric is event-type
sensitive, so a type must be emitted); the conditioning pathway is not.

**2. The two-head split is inert (+0.75, std 1.60), not beneficial as claimed.**
We justified two disjoint BIO heads by the measured 5.9 % cross-group span
overlap, arguing that a single layer would have to delete labels. The
*measurement* is right; the *inference* is wrong. Only 5.9 % of windows have a
conflict, and merging the two label spaces into one 13-type tagger gives a shared
representation that more than compensates. The per-seed spread shows the apparent
+0.75 is driven by a single lucky seed (50.10), so the honest statement is that
**the split makes no measurable difference**, not that removing it helps.

**What we report.** The frozen two-head system is the one that went through the
frozen protocol and the audit, and the differences are inside seed variance, so
the headline numbers stand. But §4 can no longer claim "four design decisions,
each forced by a measurement". Two of the four are unnecessary, and the simpler
configuration — a single merged BIO tagger plus a type-prediction head, with no
conditioning pathway — performs the same and is what we recommend.

**The methodological lesson is worth stating explicitly**, because it is the kind
of thing a paper usually hides: deriving architecture from data statistics is
seductive and was wrong here twice. The statistics (5.9 % overlap, event-type
sensitivity) were measured correctly; the architectural inferences drawn from
them did not survive a controlled test. Only ablation settles it.

**Confirmed rejections** (these behaved as the earlier, confounded table
suggested, but now on a valid single-factor comparison): CRF −0.28,
span-level role head −0.96, layer-wise LR decay −2.20.
