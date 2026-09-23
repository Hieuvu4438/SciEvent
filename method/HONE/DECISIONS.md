# DECISIONS.md — HONE

### D1. Build on CARVE, but as a component, not as the method
CARVE is our own prior method; the brief's prohibition concerns TARS-SciEvent
only. HONE vendors CARVE's tagger unchanged as a *proposer* and adds a new stage
on top. CARVE itself is left untouched in `method/SciEvent-Next` and the public
`CARVE` repository.

### D2. Direction chosen by a measured ceiling, not by intuition
Before writing the verifier we measured what a perfect verifier would reach over
CARVE's own candidates (70.9–74.2 for one seed, 80.8 for a three-seed union,
78.7 through the real decoder) against 47.1 for CARVE's threshold. No other
direction we considered had comparable measured headroom.

### D3. Keep the extraction stage segment-level
The SciEvent LLM prompts give the model "a part of a scientific abstract" — the
segment — and the tuned baselines train on windows. Using the whole abstract as
context (sibling segments) would help event typing a lot but would break
comparability, and 97 % of test documents have siblings in train. HONE uses the
window only.

### D4. Out-of-fold training candidates (cross-fitting)
The proposer memorises its training windows: in-sample candidates are 97.5 %
positive versus 32.5 % out of sample. Training a verifier on them would teach it
to keep everything. Five window-level folds stratified by event type (mirroring
the official split construction); each fold's proposer selects its epoch on the
real dev split, never on the fold it decodes. Tested explicitly as H4.

### D5. Metric-faithful labels
A candidate's target is the role of the gold span it overlaps with IoU > 0.5,
else `reject` — the same criterion the official scorer uses, so the verifier is
trained on exactly the decision the metric rewards.

### D6. Collision-aware decoding
Pooled seeds propose near-duplicates of the same gold span. Decoding visits
candidates by keep score and rejects those that collide with an accepted span;
"any overlap" vs "IoU > 0.5" is a dev-tuned choice. Gold scored spans are
non-overlapping in 97.6 % of windows, so this prior costs almost nothing.

### D7. Disk discipline
~58 GB free. Fold checkpoints (1.7 GB each) are deleted immediately after their
held-out candidates are written; frozen CARVE checkpoints are symlinked, not
copied.

### D8. Test hygiene
Dev and test candidates are produced by the frozen CARVE proposers (inference
only). The verifier, its epoch, and every decoding parameter are chosen on dev.
Test is scored once, after everything is frozen.

### D9. The window id is never an input (leakage trap found)
`sent_id` = `<paper>-<k>`, and the suffix `k` predicts the gold event type almost
perfectly: index→type diagonal accuracy 94.9 % train, **98.7 % dev**, 93.9 % test
(0→Background, 1→Methods, 2→Results, 3→Conclusions), against 88.6 % for the
text-based type head. The index is the event's position in the annotation, i.e.
annotation-derived metadata, not window text. Any model that reads it (or the
window's position in the abstract, which carries the same signal) is not solving
the benchmark's per-window task and would not be comparable to the paper's
baselines, all of which see window text only. HONE and CARVE use `sent_id` only
as a join key. Checked: no model, feature or decoding code reads it.
