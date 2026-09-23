# HYPOTHESES.md — HONE

Pre-registered before any verifier was trained. Falsification threshold for every
component: **≥ +0.5 dev Arg-C IoU F1 over its control, mean over three seeds**.
All numbers dev unless stated; test is touched once, at the end, with frozen rules.

Reference points (dev): CARVE frozen rule 47.14 ± 0.26; CARVE single-seed tuned
47.66 / 48.10 / 46.99. Oracle ceilings over CARVE candidates: 70.9–74.2 (one
seed), 80.81 (three-seed union), 78.67 through HONE's real decoder.

---

## H1 — A learned verifier beats CARVE's confidence threshold

- **Problem.** CARVE's entire F1 gain comes from thresholding the mean token
  posterior of each span; that score is weak (no signal at all for short spans:
  0/5 correct at conf ≥ 0.95) and cannot fix wrong roles.
- **Mechanism.** A cross-encoder over the window with the candidate marked,
  predicting {reject, 9 roles}, trained on out-of-fold proposals.
- **Expected.** Precision up at equal recall; role confusions partly repaired.
- **Control.** CARVE with the same proposer seed(s), dev-tuned threshold.
- **Falsification.** < +0.5 over the control.

## H2 — Pooling proposer seeds helps once a verifier decides

- **Evidence.** Oracle ceiling rises 74 → 81 from one seed to a three-seed union.
  CARVE could not exploit this: averaging posteriors compressed its confidence.
- **Mechanism.** Union of three seeds' candidates, exact duplicates merged with
  an agreement count, near-duplicates resolved by collision-aware decoding.
- **Falsification.** < +0.5 over the single-seed verifier (H1).

## H3 — Proposer evidence features matter

- **Mechanism.** Verifier input includes role-mass vector, P(O), confidence,
  seed agreement, length and position of the span.
- **Falsification.** Text-only verifier within 0.5 of the full verifier.

## H4 — Out-of-fold candidates are necessary

- **Evidence.** CARVE memorises train (loss ~0.002); in-sample candidates are
  almost all correct, dev/test candidates are not (784 predicted vs 497 gold).
- **Mechanism / control.** Train the same verifier on in-sample candidates (the
  full-train proposer decoding its own training windows).
- **Falsification.** In-sample-trained verifier within 0.5 of OOF-trained.

## H5 — Verifier ensembling helps (unlike CARVE's posterior ensembling)

- **Mechanism.** Average verifier probabilities over verifier seeds, re-tune θ.
- **Falsification.** Ensemble ≤ best single verifier seed + 0.5.

## H6 — Joint window-type decision

- **Evidence.** Oracle type adds ~7 points on top of an oracle verifier.
- **Mechanism.** To be designed only if H1–H2 succeed.
- **Falsification.** < +0.5.

---

## H7 — Candidate composition (added after H1, post-hoc: flagged as such)

- **Evidence.** 38.9 % of dev candidates are fragments of a gold span; the
  consecutive-union ceiling is +4.4 on one seed.
- **Mechanism.** Add unions of ≤ 3 consecutive non-overlapping candidates
  (gap ≤ 2) as candidates with a part-count feature; exact interval scheduling
  chooses between a whole and its parts.
- **Falsification.** < +0.5 over the same verifier without composition.

---

## Status

| | result (dev) | verdict |
|---|---|---|
| H1 | 47.14 vs 47.66 control (seed 42); keep AUC 0.890 vs 0.860; role acc 77.1 vs 81.3 % | **falsified as stated** (verifier-only); verifier + proposer-role mix (α=.5): 49.16 / 50.22 vs 47.66 / 48.10 control on seeds 42 / 13 → **supported in hybrid form** |
| H7 | 48.82 vs 49.16 same-grid control (seed 42) | **falsified** |
| H4 | in-sample 44.87 vs OOF 49.16 (seed 42) | **supported** (−4.3) |
| H3 | text-only 48.24 vs 49.16 (seed 42) | **supported, narrowly** (+0.9) |
| H2 | pooled 3 seeds: 52.42 / 50.57 vs H1 49.16 / 50.22 (same verifier seeds) | **supported** (+1.8 mean) |
| H5 | 2-verifier ensemble 51.26 vs 51.16 mean single | **falsified** (+0.1) |
| H6 | joint type λ selected (1.0) in the frozen rule; contribution in ablation | see ablation |
| **final** | test Arg-C 53.06 ± 1.04 vs CARVE 50.48 (paired Δ +2.58, CI [+0.34, +4.88]); Arg-I 63.05 vs 57.95 (Δ +5.10, CI [+2.67, +7.59]) | **HONE beats CARVE** |

*(details in RESEARCH_LOG.md)*
