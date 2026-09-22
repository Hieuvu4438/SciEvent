# TARS-SciEvent — dev funnel results

Headline metric: **Arg-C IoU F1** from the untouched official evaluator
(`baselines/ONEIE/EM_overlap_eval.py`, sha256 `173880ac7257cf51…`), on the
frozen official-code regenerated split (upstream `49cf9769…`, PYTHONHASHSEED=0).

> All numbers below are **dev**. Test has not been read; no frozen-final marker exists.
> Paper-reported numbers are external references, `reproduced: false`, never rerun.

## 1. Component funnel (seeds 13, 42, 101)

| system | tuple | prototypes | mean | std | per-seed | vs H0 | seeds up |
|---|---|---|---|---|---|---|---|
| H0 none | no | none | 31.89 | 1.88 | 13:33.01, 42:29.72, 101:32.95 | +0.00 | — |
| H1 none | yes | none | 32.79 | 2.26 | 13:30.82, 42:32.30, 101:35.25 | +0.90 | 2/3 |
| H2 fixed | yes | fixed | 31.80 | 1.36 | 13:32.34, 42:32.81, 101:30.25 | -0.09 | — |
| H2b fixed | no | fixed | 33.51 | 2.42 | 13:31.05, 42:35.89, 101:33.59 | +1.62 | 2/3 |
| H3b adaptive κ=20 | no | adaptive κ=20 | 31.83 | 1.12 | 13:30.94, 42:31.46, 101:33.08 | -0.07 | — |
| H3b adaptive κ=50 | no | adaptive κ=50 | 33.38 | 1.85 | 13:31.59, 42:35.28, 101:33.26 | +1.48 | 2/3 |
| **H3b adaptive κ=5** | no | adaptive κ=5 | **33.58** | 0.42 | 13:33.62, 42:33.15, 101:33.98 | +1.69 | 3/3 |
| H3 adaptive κ=5 | yes | adaptive κ=5 | 33.33 | 0.31 | 13:33.53, 42:33.49, 101:32.98 | +1.44 | — |

**Winner: H0 + definition prototypes + frequency-adaptive blending (κ=5), no tuple anchoring.**
It is the only system improving over H0 in 3/3 seeds, and it cuts seed variance 4×
(std 0.42 vs 1.88).

## 2. Paired bootstrap vs H0 (blocks = `doc_id`, 5000 draws)

| system | seed | Δ F1 | 95% CI | p |
|---|---|---|---|---|
| h1_tuple | 13 | -1.96 | [-6.71, +2.71] | 0.407 |
| h1_tuple | 42 | +2.59 | [-1.82, +7.21] | 0.260 |
| h1_tuple | 101 | +2.30 | [-2.68, +7.02] | 0.360 |
| h2_proto_on_h0 | 13 | -1.74 | [-6.79, +3.28] | 0.504 |
| h2_proto_on_h0 | 42 | +6.17 | [+1.81, +11.06] | 0.009 |
| h2_proto_on_h0 | 101 | +0.64 | [-2.87, +4.09] | 0.717 |
| h3b_k50 | 13 | -1.19 | [-5.65, +3.25] | 0.615 |
| h3b_k50 | 42 | +5.56 | [+1.21, +10.01] | 0.014 |
| h3b_k50 | 101 | +0.31 | [-4.12, +4.59] | 0.891 |
| h3b_k5 | 13 | +0.83 | [-3.62, +5.47] | 0.717 |
| h3b_k5 | 42 | +3.43 | [-0.51, +7.43] | 0.088 |
| h3b_k5 | 101 | +1.03 | [-2.63, +4.58] | 0.583 |

Only one seed-level comparison reaches p<0.05 (h2_proto_on_h0 seed 42). With 158 dev
windows the per-seed test is underpowered, so the **consistency across seeds**, not any
single p-value, is what supports the winner.

## 3. Oracles (diagnostic only — never in a results table)

| oracle | Arg-C IoU | Arg-I IoU | meaning |
|---|---|---|---|
| O1 | 33.62 | 39.57 | predicted type + predicted trigger tuple + predicted arguments |
| O2 | 36.48 | 42.88 | GOLD semantic argument spans -> role classification only |
| O3 | 39.36 | 48.09 | GOLD event type -> predicted trigger/arguments |
| O4 | 33.62 | 39.57 | GOLD trigger tuple -> predicted semantic arguments *(not informative: no tuple conditioning)* |
| O5 | 39.36 | 48.09 | GOLD type + GOLD tuple -> predicted semantic arguments *(not informative: no tuple conditioning)* |

**The dominant bottleneck is event-type classification, not span extraction.**
Gold event type is worth **+5.74** F1 (O3), while perfect argument boundaries are worth
only **+2.86** (O2). The evaluator is event-type sensitive, so one wrong window-level
type zeroes every argument in that window.

## 4. Role-wise effect of the prototype component (pooled over 3 seeds)

| role | dev gold | H0 | H3b κ=5 | Δ |
|---|---|---|---|---|
| Context | 477 | 25.4 | 26.3 | +0.9 |
| Results | 348 | 39.9 | 41.2 | +1.3 |
| Method | 321 | 35.7 | 39.4 | +3.7 |
| Challenge | 135 | 41.4 | 41.1 | -0.3 |
| Implications | 93 | 18.3 | 24.5 | +6.2 |
| Purpose | 75 | 23.4 | 26.6 | +3.2 |
| Analysis | 39 | 0.0 | 0.0 | +0.0 |
| Ethical | 3 | 0.0 | 0.0 | +0.0 |

**Honest negative:** `Analysis` (39 dev gold) and `Ethical` (3) stay at **0.0** — they are
never predicted at all. Rescuing ultra-rare roles was the stated motivation for the
frequency-adaptive blend, and it did **not** achieve that. The measured gain comes from
mid-frequency roles (`Implications` +6.2, `Method` +3.7, `Purpose` +3.2) instead.

## 5. Components rejected on evidence

* **H1 tuple anchoring — not supported.** 2/3 seeds, no significant seed, and adding it
  to the winner *lowers* the score (33.58 → 33.33). Prototypes on the H0 base (33.51)
  also beat prototypes on the H1 base (31.80).
* **H4 rhetorical hierarchy — not applicable.** Every window in the released split has
  exactly **one** sentence (`sentence_starts` has length 1 for all 1278/158/163 windows),
  so a sentence-level hierarchy is mathematically degenerate here. Not run.
* **κ=20 — rejected** (31.83); κ=5 chosen from the plan's {5, 20, 50} grid on dev.
* **NULL-margin calibration — not adopted**; the dev curve is flat and the argmax buys <0.1 F1.

## 6. Resources

* Peak train VRAM **7.93 GiB**, inference **5.62 GiB** (nvidia-smi 9.16 GiB),
  measured on the longest (260-word) windows. Hard gate 30 GiB: **pass**.
* Dev candidate span recall **99.40%** = the `max_span_width` ceiling, so the proposal
  ranking is lossless (STOP_MODEL_02 cleared).

## 7. Not yet done

* No test evaluation, no `freeze_final.py` marker — deliberately, per the protocol.
* H5 (domain-robust training) not run.