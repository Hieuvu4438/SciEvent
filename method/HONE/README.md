# HONE

**H**ard-negative **O**ut-of-fold **N**eural v**E**rification — propose-then-verify
argument extraction for the SciEvent benchmark (Dong et al., EMNLP 2025).

HONE is the second method in this workspace. The first, **CARVE**
(`method/SciEvent-Next`, public at `github.com/Hieuvu4438/CLAVE`), is kept
unchanged and is reused here as the *proposer*.

## Why

CARVE showed that SciEvent argument extraction is clause-level span segmentation,
and that its whole F1 gain over the paper's best baseline comes from a crude
*verifier*: a threshold on the mean token posterior of each decoded span.

We measured how good verification could be over CARVE's **own** candidates:

| | dev Arg-C IoU |
|---|---|
| CARVE's threshold (achieved) | 47.14 |
| perfect verifier, one proposer seed | 70.9 – 74.2 |
| perfect verifier, three pooled seeds | 80.81 |
| perfect verifier, through HONE's real decoder | 78.67 |

HONE replaces the threshold with a learned verifier.

## How

```
window ──► CARVE tagger × 3 seeds ──► pooled candidate spans (recall-oriented, no threshold)
                                         │
                                         ▼
      event type: <T> . w1 … <a> span </a> … wn   +  proposer evidence (16 features)
                                         │
                                  DeBERTa-v3-large cross-encoder
                                         │
                                         ▼
                        {reject, Context, Method, Results, …}   per candidate
                                         │
                                         ▼
                 keep-score threshold + collision-aware selection  ──► arguments
```

**Out-of-fold training candidates.** CARVE memorises its training windows: the
candidates it produces on them are 97.5 % correct, against 32.5 % out of sample.
A verifier trained on those would learn to keep everything. HONE cross-fits the
proposer over five folds so every training window gets candidates from a model
that never saw it; out-of-fold candidates match dev on volume (≈5 per window) and
error rate (≈36 % vs 39.6 % positive).

## Status

**Research complete; archived.** The released method is **CLAVE** = CARVE-simple + HONE
(one proposer seed, one verifier seed), shipped in `github.com/Hieuvu4438/CLAVE`.
Test: Arg-C IoU 52.36, Arg-I IoU 60.09, trigger ROUGE-L 77.82
(full metrics: `CARVE_SIMPLE_REPORT.md`).

Other configurations developed here, with their own frozen dev rules and single test looks:

| configuration | test Arg-C | test Arg-I |
|---|---|---|
| HONE on 3 pooled original-CARVE proposers, 3 verifiers | 53.06 ± 1.04 | 63.05 ± 0.27 |
| HONE on the original-CARVE proposer, 1 seed | 53.35 | 60.15 |
| **CARVE-simple + HONE, 1 seed (released)** | **52.36** | **60.09** |

The full record is in `RESEARCH_LOG.md` (stages 0–13, including the test-look
disclosure), `HYPOTHESES.md` and `DECISIONS.md`.

## Layout

```
hone/          proposer (vendored CARVE tagger) + candidates, verifier, select
scripts/       propose.py (eval / oof / insample), train_verifier.py, freeze_and_test.py
configs/       proposer.json (the frozen CARVE recipe)
data/          folds and candidate files          (not tracked)
runs/          checkpoints and logs               (not tracked)
```
