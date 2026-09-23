# SciEvent — research archive

Methods, experiment records and results for argument extraction on the
**SciEvent** benchmark (Dong et al., EMNLP 2025). The benchmark itself is a
read-only submodule in `third_party/SciEvent`.

**The final method is maintained separately:**
[github.com/Hieuvu4438/CARVE](https://github.com/Hieuvu4438/CARVE), which ships **CARVE-simple + HONE**.
This repository keeps the development of every method, including the ones that were superseded.

## Methods

| folder | method | status |
|---|---|---|
| [`method/HONE`](method/HONE) | **HONE**: propose-then-verify with an out-of-fold cross-encoder verifier. Includes the development of the released **CARVE-simple + HONE** | archived development record of the final method |
| [`method/CARVE-full`](method/CARVE-full) | **CARVE**, original release: two BIO heads, event-type conditioning and a calibrated threshold, with its single-factor ablations | superseded; previous content of the CARVE repository |
| [`method/SciEvent-Next`](method/SciEvent-Next) | development workspace in which CARVE was built | archived |
| [`method/TARS-SciEvent`](method/TARS-SciEvent) | earlier approach | archived |

## Results (test, official scorer)

| system | seeds | ROUGE-L F1 | Arg-I IoU F1 | Arg-C IoU F1 |
|---|---|---|---|---|
| OneIE (best published, Arg) | — | 72.40 | 53.57 | 41.61 |
| GPT 5-shot (best published, ROUGE-L) | — | 75.08 | 49.98 | 34.47 |
| CARVE (original) | 3 | 76.93 ± 0.80 | 57.95 ± 2.46 | 50.48 ± 1.15 |
| CARVE-simple | 3 | 77.22 ± 0.81 | 58.58 ± 0.57 | 50.73 ± 0.19 |
| HONE, original-CARVE proposer | 1 | 76.02 | 60.15 | 53.35 |
| HONE, 3 pooled original-CARVE proposers | 3 verifiers | 78.05 | 63.05 ± 0.27 | 53.06 ± 1.04 |
| **CARVE-simple + HONE (released)** | 1 | **77.82** | **60.09** | **52.36** |

- Every system froze its decoding rule on dev before its single test evaluation. Several test looks happened during development; all of them are disclosed in `method/HONE/RESEARCH_LOG.md` (stage 13).
- Precision/recall breakdowns are in `method/HONE/CARVE_SIMPLE_REPORT.md`.

## Where to read

- `method/HONE/CARVE_SIMPLE_REPORT.md` — full metrics for CARVE-simple and CARVE-simple + HONE.
- `method/HONE/RESEARCH_LOG.md`, `HYPOTHESES.md`, `DECISIONS.md` — the HONE research process: pre-registered hypotheses, and what was supported or falsified.
- `method/CARVE-full/docs/` — the original CARVE paper notes and write-up.
