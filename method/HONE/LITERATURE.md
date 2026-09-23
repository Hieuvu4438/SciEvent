# LITERATURE.md — HONE

HONE is the second method in this workspace. The first, CARVE (`method/SciEvent-Next`,
public at `github.com/Hieuvu4438/CLAVE`), established that SciEvent argument
extraction is clause-level span segmentation and reached 50.48 Arg-C IoU on test.
Its full literature review is in `method/SciEvent-Next/LITERATURE.md`; this file
covers only what is new for HONE.

## 0. Is there a newer comparable result?

Searched again (Sept 2026). The only paper after SciEvent that reports on the
Dong et al. benchmark is CARVE itself. **EXCEEDS** (Lu et al., ACL 2026,
arXiv 2406.14075) is *not* comparable: it evaluates on **SciEvents**, a different
dataset the authors built (2,508 ACL abstracts, 24,381 events), with its own
schema. This is exactly the "similarly named dataset" trap the research brief
warns about.

**Targets for HONE:** CARVE 50.48 Arg-C IoU (test) / 47.14 ± 0.26 (dev, frozen
rule); the paper's best is OneIE 41.61.

## 1. The measurement that selects the direction

CARVE's own decomposition showed that *all* of its F1 gain over OneIE came from a
crude verifier: a threshold on the mean token posterior of each decoded span.
Uncalibrated CARVE scores 39.89 (below OneIE); thresholding lifts it to 50.48.

So the question for a second method is how good verification *could* be. We
measured it before building anything, by passing CARVE's own argmax candidates
through an oracle verifier (keep a candidate iff it IoU>0.5-matches gold; relabel
it with the gold role):

| candidate set (dev) | oracle Arg-C IoU | + oracle window type |
|---|---|---|
| one CARVE seed (~800 candidates) | 70.9 – 74.2 | 77.6 – 80.2 |
| union of 3 CARVE seeds (1,764) | **80.81** | 88.44 |
| same union, through HONE's real decoder | **78.67** | — |
| CARVE's threshold (what is achieved today) | 47.14 | — |

A 30-point gap between the heuristic verifier and a perfect one, *over the same
candidates*. No other direction we could identify has comparable headroom, so
HONE targets verification.

## 2. Related work and what transfers

**SpanNER** (Fu, Huang & Liu, ACL 2021) — span prediction as a *system combiner*:
a span classifier re-recognises entities from the pooled outputs of many NER
systems. The closest idea to HONE's pooled-proposals-plus-verifier. Differences:
SpanNER's combiner is trained on gold/enumerated spans, not on the base systems'
own out-of-sample mistakes; its targets are short entity mentions; and it has no
notion of the calibration gap that motivates HONE.

**Locate and Label** (Shen et al., ACL 2021) — two-stage nested NER: propose
spans (filtering + boundary regression), then label them. Transfers the
propose-then-label decomposition. HONE does not regress boundaries: CARVE's
length bias on matched spans is ~0 and IoU>0.5 already tolerates small boundary
error, so boundary refinement has little headroom here.

**Entity markers / PURE** (Zhong & Chen, NAACL 2021) — wrapping a span in marker
tokens and reading the marker representations is a strong, simple span encoder.
HONE's verifier uses exactly this encoding.

**Rerankers trained on the base model's own errors** — rerankers in semantic
parsing and QA commonly harvest negatives from a generator's intermediate
predictions "so that the candidate distribution in training matches validation".
HONE applies the same principle, made necessary rather than optional by a
specific property of the proposer (next section).

**Stacked generalisation / cross-fitting** (Wolpert 1992; out-of-fold prediction
in stacking) — the standard remedy when a second-level model must be trained on
a first-level model's predictions without the first level having seen those
examples. HONE uses K-fold cross-fitting to produce training candidates.

**Hard-negative mining for cross-encoder rerankers** — hard negatives drawn close
to the positives are what teach fine discrimination. HONE's negatives are the
proposer's own near-misses (fragments, wrong-role spans, spurious clauses), which
are as hard as negatives can be for this task.

## 3. Why out-of-fold candidates are not optional here

CARVE's training loss reaches ~0.002: it memorises its 1,278 training windows.
Candidates it produces *on its own training windows* are therefore almost all
correct, and a verifier trained on them would learn "keep everything". Dev/test
candidates, by contrast, are full of errors (784 predicted vs 497 gold on dev).
Training and test candidates would come from different distributions.

Cross-fitting removes this: each train window's candidates come from a proposer
that never saw it. Folds are window-level and stratified by event type, mirroring
the official split, so held-out windows keep sibling segments in the proposer's
training data exactly as dev/test windows do. This is testable, and HONE tests it
(H4 in HYPOTHESES.md): train the verifier on in-sample candidates instead and
measure the loss.

## 4. Ranked opportunity map for HONE

1. **Learned verifier over CARVE candidates** (H1) — largest measured headroom.
2. **Pool candidates across proposer seeds** (H2) — raises the ceiling 74 → 81.
3. **Proposer evidence as verifier features** (H3) — lets the verifier learn when
   to trust the proposer; cheap.
4. **Out-of-fold vs in-sample candidates** (H4) — the claim that justifies the
   design; must be demonstrated, not asserted.
5. **Verifier ensembling** (H5) — CARVE's posterior ensembling failed because it
   compressed the confidence the threshold relied on; a verifier trained with CE
   may behave differently. Test, do not assume.
6. **Joint window-type decision** (H6) — oracle type is still worth ~7 points on
   top of an oracle verifier; the proposer's type head is the remaining ceiling.

Sources: [SpanNER](https://aclanthology.org/2021.acl-long.558/),
[Locate and Label](https://aclanthology.org/2021.acl-long.216/),
[EXCEEDS](https://aclanthology.org/2026.acl-long.271/),
[SciEvent](https://aclanthology.org/2025.emnlp-main.871/).
