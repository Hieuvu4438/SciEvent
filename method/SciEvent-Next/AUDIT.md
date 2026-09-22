# AUDIT.md — leakage and protocol verification

Everything below was run after the frozen test evaluation, to check whether the
result is real. Reproduce with `bash scripts/audit.sh`.

---

## A. Do our numbers reproduce under the *official* evaluator, run as a CLI?

We import the upstream scorer as a library, so the first thing to rule out is a
wrapper bug. Running the upstream script exactly as the benchmark README
prescribes, on our frozen test predictions:

```
python3 third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py \
  --pred  method/SciEvent-Next/artifacts/preds/test_preds_s13.jsonl \
  --gold  third_party/SciEvent/SciEvent_data/ONEIE/all_splits/test.oneie.json
```

```
[ROUGE-L OVERALL] P: 84.68%, R: 76.77%, F1: 77.52%
Argument Identification (Exact)  - P: 41.26 (184/446)  R: 34.52 (184/533)  F1: 37.59
Argument Classification (Exact)  - P: 37.67 (168/446)  R: 31.52 (168/533)  F1: 34.32
Argument Identification (IoU)    - P: 65.92 (294/446)  R: 55.16 (294/533)  F1: 60.06
Argument Classification (IoU)    - P: 56.50 (252/446)  R: 47.28 (252/533)  F1: 51.48
```

**Identical to our reported seed-13 numbers, to the last decimal.** No wrapper bug.

Additionally, `tests/test_contract.py` routes *gold* spans through our own
prediction writer into the official evaluator and scores **100.00** on every
metric, on both dev and test — so the export format itself is exact.

## B. Window-level leakage — none

| overlap | count |
|---|---|
| train ∩ test windows | **0** |
| train ∩ dev windows | **0** |
| dev ∩ test windows | **0** |
| exact window-text duplicates train → test | **0** |

## C. Document-level overlap — a property of the benchmark, not of our method

This is the one real finding of the audit.

The paper describes an 80/10/10 split **by document**. The released
`data_scripts/shared/split_data.py` in fact dedups and splits on **`wnd_id`**
(window), stratified by event type — `key = item["wnd_id"]`. The consequence:

| | value |
|---|---|
| test documents | 147 |
| test documents that also appear in train | **143 (97.3 %)** |
| test documents unseen in train *and* dev | 2 |

So the *windows* are disjoint but the *abstracts* are not: the Background segment
of a paper can be in train while its Results segment is in test.

**This does not invalidate the comparison.** OneIE, DEGREE and EEQA were all
trained and scored on these exact files, per the benchmark README, so every
number in the paper's Table 3 and Table 4 carries the same property. The
comparison in `FINAL_REPORT.md` is apples-to-apples.

It does mean the official test number alone cannot say how much of the score
comes from abstract familiarity. §D measures that directly.

## D. Document-disjoint re-split — how much of the score survives?

`scripts/make_docsplit.py` rebuilds a split with the same proportions and the
same event-type stratification, but keyed on `doc_id`, so no abstract crosses a
boundary (394 / 45 / 61 documents → 1265 / 148 / 186 windows, verified zero
document overlap). The identical frozen recipe is retrained on it.

The identical protocol was applied: 3 seeds, decoding rule re-frozen on the *new*
dev only (`freeze_rules.py docsplit ...` → τ = 0.85, min_len = 3, dev Arg-C
47.79 ± 1.64), then one evaluation on the new test.

| | Arg-I IoU F1 | **Arg-C IoU F1** | ROUGE-L F1 |
|---|---|---|---|
| ours, official split | 57.95 ± 2.46 | **50.48 ± 1.15** | 76.93 ± 0.80 |
| ours, **document-disjoint** | 52.27 ± 1.43 | **44.93 ± 0.95** | 76.20 ± 1.00 |
| OneIE / GPT-5shot (official split only) | 53.57 | 41.61 | 75.08 |

**Arg-C stays above the paper's best even with zero document overlap** (44.93 vs
41.61). Arg-I falls slightly below it (52.27 vs 53.57), and ROUGE-L stays above
(76.20 vs 75.08).

### Decomposing the 5.55-point drop

The re-split is **harder by construction**: its test half is 33.7 % Digital
Humanities gold arguments against 21.0 % in the official test split, and DH is by
far our weakest domain. Comparing domain by domain separates the two effects:

| domain | official F1 (n) | docsplit F1 (n) | Δ |
|---|---|---|---|
| ACL | 59.21 (60) | 51.10 (63) | −8.11 |
| bioinfo | 74.39 (52) | 52.96 (57) | −21.43 |
| cscw | 48.61 (91) | 51.35 (86) | **+2.75** |
| dh | 29.75 (112) | 25.88 (189) | −3.87 |
| jmir | 52.11 (218) | 52.97 (166) | **+0.86** |

Reweighting the document-disjoint per-domain scores to the *official* domain mix:

| | Arg-C IoU F1 |
|---|---|
| docsplit, its own DH-heavy mix | 43.38 |
| docsplit, **reweighted to the official mix** | **46.79** |
| official split, same weighting | 49.79 |

So of the 5.55-point total drop, roughly **3.4 points come from the DH-heavy test
mix** and roughly **3.0 points from removing document overlap**.

**Conclusion.** Document overlap is worth about **3 Arg-C F1** to our method — real,
but far short of the +8.87 margin. Two domains (cscw, jmir) actually *improved*
under the stricter split, which is not what systematic leakage looks like.

**Important asymmetry, stated plainly:** OneIE was never re-measured on the
document-disjoint split, and it would very likely drop too. Comparing our
document-disjoint 44.93 against OneIE's official-split 41.61 is therefore
*unfavourable to us*, not favourable. The headline claim in `FINAL_REPORT.md`
rests on the official split, where the comparison is genuinely apples-to-apples;
this section exists only to show the margin is not an artifact of the benchmark's
split construction.

## E. Could the training code have seen test?

Every `load_split` call in the training path:

```
train.py:218   load_split("train", ...)
train.py:219   load_split(cfg.get("eval_split", "dev"), ...)
freeze_rules.py:35   load_split("dev")
```

`eval_split` is unset in **all six** configs, so it defaults to `dev`. The
training loop is structurally incapable of reading test.

## F. Was anything tuned on test?

The frozen decoding rule was written **before** any test prediction existed:

```
2026-09-22 21:35:50  artifacts/decoding_rules.json
2026-09-22 21:36:05  artifacts/preds/test_preds_s42.jsonl
2026-09-22 21:36:14  artifacts/preds/test_preds_s13.jsonl
2026-09-22 21:36:24  artifacts/preds/test_preds_s101.jsonl
```

τ and `min_len` were selected by `scripts/freeze_rules.py`, which maximises the
**mean dev Arg-C IoU across seeds** — chosen that way specifically so the rule is
not fitted to whichever seed looks best. Test was evaluated once per seed with
that one rule, and no rule was changed afterwards.

### Full disclosure: every time test data was touched during development

1. `tests/test_contract.py` — uses test **gold labels** to verify the exporter
   scores 100.00. No model involved, so no model-selection signal.
2. A one-off check of the **gold span-length distribution** on test (6.8 % of
   spans are ≤2 tokens), run to sanity-check whether the `min_len` rule would
   transfer. `min_len = 3` had **already** been selected by the dev grid search
   before this was computed, and the final frozen value came from
   `freeze_rules.py` on dev alone. It changed nothing, but it was looked at and
   is disclosed here rather than omitted.
3. A tokenizer length check (max 240 sub-tokens on test) — token counts only.
4. `scripts/make_docsplit.py` pools all three official splits to build the
   document-disjoint re-split in §D. That re-split is therefore *not* comparable
   to the paper and is reported only as a robustness probe.
5. The single frozen evaluation itself.

## G. Are we exploiting the evaluator rather than solving the task?

| check | result |
|---|---|
| predicted windows vs gold windows | 163 / 163, ids match exactly |
| events per predicted window | exactly 1 |
| predicted scored arguments vs gold | 446 vs 533 — **ratio 0.84**, we *under*-predict |
| duplicate `(span, role)` predictions | **0** |
| gold event type used at inference | **no** — type is predicted (91.41 % accurate) |
| evaluator source modified | **no** — `third_party` has 0 modified files |

The duplicate check matters because `compute_f1` accumulates matches into a
`set` of `(pred, gold)` pairs; emitting identical spans repeatedly could
interact with that. We emit none. The under-prediction ratio matters because it
rules out winning precision by flooding the output.

## H. Is "exactly one event per window" a test-derived assumption?

No. It holds on **all 1278 training windows**, so the assumption is derivable
from training data alone. It is a modelling choice that exploits the structure of
the task, available to any system trained on the same data, and it is stated
explicitly in `FINAL_REPORT.md` §4.

## I. Known non-leakage caveats about the comparison

- **Segmentation is not modelled.** We consume the provided windows, as all three
  tuned baselines do. We report no segmentation number rather than an
  incomparable one.
- **Our operating point favours precision.** Against OneIE we are ahead on
  Arg-C precision (+15.61) and recall (+2.76), but on **Arg-I our recall is
  2.94 lower** than OneIE's (53.35 vs 56.29) while precision is 12.35 higher.
  The same holds for trigger ROUGE-L (recall −2.54, precision +10.15). This is
  the τ = 0.90 threshold doing exactly what it was tuned to do — maximise
  **Arg-C IoU F1**, the paper's primary metric. It is a single dial and the dev
  curve for it is in `RESEARCH_LOG.md`; a recall-oriented operating point exists
  but was not evaluated on test, because the rule was frozen first.
- **The segment ordinal is deliberately unused** (see `FINAL_REPORT.md` §7),
  though it would be legitimate for a deployed system and is worth ~+4.7 Arg-C.

## J. Is the confidence threshold "gaming the metric"?

The fairest challenge to this result is not leakage — it is that most of the
margin comes from a decoding threshold rather than from the model. That is worth
answering directly rather than burying.

**The decomposition (test, mean over 3 seeds):**

| stage | Arg-C P | Arg-C R | Arg-C F1 |
|---|---|---|---|
| OneIE | 39.69 | 43.71 | 41.61 |
| ours, uncalibrated argmax | 32.43 | **51.83** | 39.89 |
| ours, calibrated (τ = 0.90) | 55.30 | 46.47 | **50.48** |

Three observations:

1. **Uncalibrated, our model scores below OneIE** (39.89 vs 41.61). We do not
   claim the reframing alone beats the baselines.
2. **What the reframing buys is recall** — 51.83 against OneIE's 43.71. That is a
   real +8.12 in gold arguments actually found, before any threshold exists. The
   threshold cannot create recall; it can only trade it away.
3. The threshold therefore converts a *pre-existing* recall advantage into F1.
   A baseline without that recall headroom could not reach 50.48 by thresholding.

**On legitimacy.** τ, `min_len` and `merge_gap` are three scalars fitted on the
development split and frozen before test — the same class of choice as a learning
rate or a checkpoint-selection criterion. No test statistic informed them
(§F), the dev optimum is a broad plateau rather than a spike
(`RESEARCH_LOG.md`), the rule was chosen to maximise the **mean across seeds**
rather than any single seed, and test came out *above* dev (50.48 vs 47.14),
which is the opposite of what dev-overfitting looks like.

**What would be illegitimate**, and was not done: tuning τ on test, selecting the
checkpoint on test, or reporting the best of several test evaluations. Test was
evaluated once per seed with one rule.

**What is a genuine limitation**, stated plainly: OneIE was reported at its own
natural operating point, and we report at a tuned one. A precision/recall-matched
comparison would narrow the Arg-C gap. Our Arg-I recall (53.35) is in fact
*below* OneIE's (56.29) for exactly this reason. The honest summary is that the
method wins clearly on F1 across every matching mode, and wins on recall only
before calibration.
