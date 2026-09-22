# DECISIONS.md — SciEvent-Next

Design decisions, each with the evidence that forced it.

### D1. Do not start from TARS-SciEvent, and do not reproduce baselines
Per the research brief. `method/TARS-SciEvent` was excluded from all searching and
reading. Paper-reported baseline numbers are used as targets directly. No time was
spent on OneIE/DEGREE/EEQA reproduction.

### D2. Measure the data before reading the EAE literature
The measurement (`LITERATURE.md` §0) changed the problem class from
mention-style EAE to clause-level span segmentation. Reading the literature first
would have biased us toward generative/QA formulations that the data contradicts.

### D3. Consume gold event segments
All tuned baselines in the paper (OneIE, DEGREE, EEQA) are trained and scored on
the provided windows; the segmentation numbers in the paper are a separate
zero-shot LLM experiment. We use the same `*.oneie.json` windows, which is the
directly comparable protocol. Segmentation is therefore out of scope, and we say
so rather than reporting an incomparable end-to-end number.

### D4. Use the official evaluator as a library, never a reimplementation
`evaluate.py` `exec`s `EM_overlap_eval.py` and calls its own functions. This makes
metric drift impossible. Verified by an oracle test scoring exactly 100.00.

### D5. Two disjoint BIO heads, not one
Measured overlap: within scored roles 2.4 %, within Agent/PrimaryObject/
SecondaryObject 0.2 %, **across the two groups 5.9 %**. A single head would have
to resolve that 5.9 % by deleting labels; two heads do not. The AAO head exists
only to produce the ROUGE-L tuple, which the Arg-I/Arg-C metrics ignore.

### D6. BIO tagging rather than span enumeration
The BIO ceiling was measured before committing: 99.70 Arg-C IoU on dev. Span
enumeration would recover the remaining 0.3 % at the cost of O(n²) candidates and
a harder precision problem. Revisit only if the model gets within 1 F1 of ceiling.

### D7. Shorter spans win BIO conflicts
When two gold spans overlap, the shorter one is written first and the longer is
truncated to its remaining contiguous run. Rationale: the shorter span is the more
specific annotation, and a truncated long span often still clears IoU > 0.5.
Empirically this recovers 98.97 % of gold role spans exactly.

### D8. Event type is modelled explicitly and conditions the role head
The metric is event-type sensitive with one event per window, so a wrong window
type zeroes every argument in it, as both false positives and false negatives.
Conditioning uses a learned type embedding added to word states, with scheduled
sampling from gold to self-prediction so training and inference agree.

### D9. Discriminative first, generative only as a fallback
DEGREE's reported Arg-C IoU recall on this exact benchmark is 13.83 — the
signature of generative copying failing on 10–15-token spans. A 0.4B encoder
trains in 13 minutes; a 7B LoRA costs hours per hypothesis. Information gain per
GPU-hour is far higher on the discriminative side.

### D10. fp32 master weights
`transformers>=5` honours the dtype stored in a checkpoint, and
`microsoft/deberta-v3-large` ships fp16 weights. AdamW on fp16 parameters produced
NaN on the **first** optimizer step (diagnosed: gradients finite, gradient norm
11.02, all 390 parameter tensors NaN after `opt.step()`). The encoder is loaded
with `dtype=torch.float32`; mixed precision comes from autocast only.

### D11. Dev/test hygiene
All development uses train + dev only. Test is untouched until the architecture,
checkpoint-selection rule, decoding rules, and any thresholds are frozen. The only
prior contact with test is the oracle correctness test, which uses gold labels to
verify the exporter and produces no model-selection signal.
