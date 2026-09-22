# TARS-SciEvent — agent operating notes

Source of truth for the protocol: `docs/proposal1/SCIEVENT_TARS_END_TO_END_AGENT_PLAN.md`.
This file records what the implementation actually does and the decisions that
the data forced.

## Hard rules

1. **Do not train or reproduce OneIE / DEGREE / EEQA.** `vendor/SciEvent/baselines/`
   is read only for the official evaluator and data converters.
2. **Never tune on test.** `scripts/train_tars.py --final-test` and
   `scripts/predict_tars.py --split test` both refuse to run until
   `artifacts/final/FROZEN_BEFORE_TEST.txt` exists.
3. **The official evaluator is the headline metric.** We never reimplement
   scoring and assert equivalence. `official_wrapper.py` validates the
   prediction file, then subprocesses the untouched
   `vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py`.
4. **Prediction ID set == gold ID set**, always. Upstream silently ignores
   predictions under an unknown `sent_id` (proved in
   `tests/test_metric_contract.py::test_unknown_pred_sent_id_is_ignored_upstream`),
   so a missing assert would let us benefit from an evaluator artifact.
5. **Architectural capacities come from TRAIN only.** `registry.resolve_capacities`
   derives them and writes both the value and the rule into the run manifest.

## What the released data actually looks like

Measured from `data/official/` (see `data/manifests/split_audit.json`):

| property | train | dev | test |
|---|---|---|---|
| windows | 1278 | 158 | 163 |
| events per window | **exactly 1** | exactly 1 | exactly 1 |
| max subword length | 618 | 285 | 300 |
| semantic arg width p99.5 / max | 40 / 76 | 42 / 44 | 41 / 46 |
| max same-role args per event | 17 (Results) | 10 | 13 |

Consequences, all derived from TRAIN:

* `K_event = 2` (`max_train_events_per_window + 1`). The event-set decoder is
  kept even though it is nearly degenerate here — flattening would violate rule 6
  of the plan and would not generalise.
* `max_length = 768`, not the plan's suggested 512: one train window needs 618
  subwords and **no gold span may be silently truncated**.
* `max_span_width = 40` = `min(64, ceil(p99.5))`. Recall ceiling 99.5% train /
  99.4% dev, both above the STOP_MODEL_02 gate.
* `k_start = k_end = 128`, `k_span = 2048`. The plan's starting value
  `k_span = 512` capped dev candidate recall at **97.6%**, below the 98% gate,
  and the recall *fell* during training as the boundary head sharpened. The
  sweep (`artifacts/audits/candidate_recall_sweep.json`) shows `k_span` was the
  binding constraint, not the boundary top-K: at 2048 the proposal ranking is
  lossless (dev recall 99.40% == the width-cap ceiling), so any remaining miss
  is the width cap alone. Cost in dev Arg-C IoU: none beyond seed noise
  (33.65 -> 33.01 on seed 13, against a ~2 point seed spread).
* Per-role slot counts = train max + 1 → 78 role slots per event.
* Component slots: Agent 1, PrimaryObject 1, SecondaryObject 4 (train maxima).

**Label names differ from the plan's shorthand.** The released files use
`Background/Introduction`, `Methods/Approach`, `Results/Findings`,
`Conclusions/Implications`, and the roles `Results`, `Implications`,
`Contradictions` (plural). The evaluator compares raw strings, so
`data/schema.py` uses the released names.

## Two normalisation decisions that mattered

* The argument loss is normalised **per gold argument**, not per role slot. With
  78 slots and ~3 real arguments per event, slot-averaging diluted the signal
  ~78×; `null_weight = 0.1` (DETR convention) keeps the dominant NULL slots from
  swamping it.
* The boundary proposal BCE uses `pos_weight = 5.0`; gold boundaries are ~5% of
  words, and candidate recall is a hard gate.

`force_min_events: 1` emits the best slot when every slot predicts NONE. This is
a **train-derived prior** (every train window has ≥1 event), never the gold count
of the example being decoded.

## Component funnel

Each config changes exactly one thing relative to its parent:

```
h0_span.yaml            encoder + event set + candidates + learned role vectors
h1_tuple.yaml           + q_tuple conditioning                 (use_tuple_query)
h2_proto.yaml           + fixed codebook definition prototypes (prototype_mode: fixed)
h3_adaptive_proto.yaml  + lambda_r = kappa/(kappa+n_r)         (prototype_mode: adaptive)
h4_rhetoric.yaml        + rhetorical sentence hierarchy        (use_discourse)
```

Promote a component only when dev Arg-C IoU improves in ≥2 seeds, with no
material Arg-C EM collapse, not from one role/domain, and VRAM ≤30 GiB.
A negative H1 must be reported as negative, not dropped silently.

## Commands

```bash
bash scripts/00_preflight.sh                        # audit + protocol gate + tests
python scripts/train_tars.py --config configs/tars/h0_span.yaml --seed 42 --smoke-test --overfit
python scripts/train_tars.py --config configs/tars/h0_span.yaml --seed 42
python scripts/predict_tars.py --config ... --checkpoint ... --split dev --breakdowns
python scripts/run_oracles.py --config ... --checkpoint ... --split dev
python scripts/tune_candidates.py --config ... --checkpoint ... --split dev
python scripts/calibrate_null.py --config ... --checkpoint ...
python scripts/measure_vram.py --config configs/tars/h0_span.yaml
python scripts/freeze_final.py --config <winner> --dev-evidence artifacts/runs/<id>/manifest.json
python scripts/final_report.py --runs artifacts/runs --split dev
```

Any config value can be overridden without editing files:
`--override training.backbone_lr=2e-5 --override model.k_span=256`.

## Superseded runs

`artifacts/runs_superseded/` holds runs whose configuration was later replaced
(currently the `k_span=512` H0 runs that failed the recall gate). They stay out
of `artifacts/runs/` so `final_report.py` can never mix two proposal capacities
into one results row.

## Oracles

`O4`/`O5` pin the trigger tuple to gold. In H0 the argument decoder is
conditioned on the raw event-slot state, so these oracles **cannot** change the
semantic arguments; `evaluation/oracles.py` reports them as not informative
rather than printing a misleading number. They become meaningful from H1 on.

Oracle results never enter a results table.

## Comparison policy

`artifacts/reference/paper_reported_results.json` holds published numbers with
`reproduced: false` — they are reference points, never rerun, and
`final_report.py` re-stamps the flag on every run. Paired bootstrap significance
is computed **only** between our own systems (document-block resampling by
`doc_id`), never against a paper-only number.
