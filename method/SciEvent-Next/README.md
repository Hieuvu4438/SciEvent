# SciEvent-Next

An independent method for the **SciEvent** benchmark (Dong et al., EMNLP 2025).

## Result

Frozen evaluation on the official test split, 3 seeds, official evaluator:

| metric | paper best | **SciEvent-Next** | Δ |
|---|---|---|---|
| **Arg-C IoU F1** (primary) | 41.61 (OneIE) | **50.48 ± 1.15** | **+8.87** |
| **Arg-I IoU F1** | 53.57 (OneIE) | **57.95 ± 2.46** | **+4.38** |
| **Trigger ROUGE-L F1** | 75.08 (GPT 5-shot) | **76.93 ± 0.80** | **+1.85** |

Every individual seed beats every target. See `FINAL_REPORT.md`.

This directory is self-contained. It does not read from, extend, or share code
with any other method in `method/`. It treats `third_party/SciEvent` as
**read-only**: the datasets and the official evaluator are imported, never modified.

---

## The idea in one paragraph

Every tuned baseline in the SciEvent paper (OneIE, DEGREE, EEQA) models arguments
the way ACE-style event extraction does: short entity mentions, found by graph
decoding, template generation, or span-QA. We measured the benchmark first, and
the data says something different. Each annotated window contains **exactly one
event**; the scored arguments are **10–15-token clauses**; **97.6 %** of windows
have no overlap at all between scored arguments; and the most common gap between
two consecutive arguments is **zero tokens**. SciEvent argument extraction is
therefore **clause-level span segmentation with semantic-role labelling**, not
mention extraction. SciEvent-Next models it that way: a shared encoder with a
word-level BIO head over the nine scored roles, a second disjoint BIO head for the
⟨Agent, Action, Object⟩ trigger tuple, and a window-level event-type head whose
prediction conditions the role head (the official metric is event-type sensitive
and trigger-insensitive, so window type acts as a multiplier on every argument).

See `LITERATURE.md` for the evidence, `HYPOTHESES.md` for the falsifiable backlog,
and `RESEARCH_LOG.md` for every run.

## Metric contract

We never reimplement the metrics. `src/scievent_next/evaluate.py` loads
`third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py` and calls **its own
functions**, so matching semantics are bit-identical to the paper's numbers:
event-type-sensitive, trigger-insensitive, `{Agent, PrimaryObject,
SecondaryObject}` excluded from Arg-I/Arg-C, IoU > 0.5, and the ROUGE-L score
computed over the ⟨Agent, trigger, PrimaryObject, SecondaryObject⟩ tuple.

`tests/test_contract.py` proves the export is exact: gold spans routed through our
own prediction writer into the official evaluator score **100.00** on every metric.

## Data

Already present, nothing to download:

```
third_party/SciEvent/SciEvent_data/ONEIE/all_splits/{train,dev,test}.oneie.json
```
1278 / 158 / 163 windows. Discovery and validation:

```bash
python3 tests/test_contract.py
```

## Install

```bash
pip install -r requirements.txt
```

## Train and evaluate

Everything, end to end (~35 min on one RTX 5880 Ada):

```bash
bash scripts/reproduce.sh
```

Or step by step:

```bash
python3 tests/test_contract.py                                  # correctness + evaluator oracle
python3 src/scievent_next/train.py -c configs/final.json --set run_name=my_run seed=42
python3 scripts/freeze_rules.py                                 # freeze decoding on DEV only
python3 src/scievent_next/infer.py --ckpt artifacts/runs/my_run/best.pt \
    --split test --rules artifacts/decoding_rules.json --out artifacts/preds/test.jsonl
python3 src/scievent_next/diagnose.py --pred artifacts/preds/test.jsonl --split test
```

Any config key can be overridden inline:

```bash
python3 src/scievent_next/train.py --set run_name=my_run seed=13 epochs=40
```

## Layout

```
src/scievent_next/data.py      data contract, label inventories, BIO encode/decode
src/scievent_next/model.py     encoder + role / AAO / event-type heads
src/scievent_next/train.py     training loop, checkpoint selection, prediction export
src/scievent_next/evaluate.py  thin wrapper over the OFFICIAL upstream scorer
src/scievent_next/infer.py     posterior caching, decoding rules, frozen evaluation
src/scievent_next/crf.py       linear-chain CRF (tested, falsified - see HYPOTHESES.md H4)
src/scievent_next/diagnose.py  error taxonomy + per-role/domain/type/length tables
tests/test_contract.py         data, BIO round-trip, and evaluator oracle tests
configs/final.json             the frozen recipe; other configs are the ablations
scripts/reproduce.sh           end-to-end pipeline
scripts/freeze_rules.py        dev-only decoding-rule selection
scripts/registry.py            regenerates artifacts/EXPERIMENTS.md
scripts/full_tables.py         regenerates the Table 3 / Table 4 comparisons
scripts/audit.sh               reproduces the leakage / protocol audit
scripts/make_docsplit.py       document-disjoint re-split (leakage robustness probe)
artifacts/runs/<run>/          log.json, dev_preds.jsonl, best.pt
artifacts/decoding_rules.json  the frozen decoding rule (tau=0.90, min_len=3)
artifacts/preds/               frozen test predictions, one file per seed
```

## Documents

| file | contents |
|---|---|
| `FINAL_REPORT.md` | the full write-up: findings, method, results vs both paper tables, negative results, weaknesses, next steps |
| `AUDIT.md` | **leakage + protocol audit**: official-CLI re-verification, split overlap, freeze ordering, full disclosure of every test touch |
| `LITERATURE.md` | literature review and ranked opportunity map |
| `HYPOTHESES.md` | the falsifiable backlog, each with its result and keep/reject decision |
| `RESEARCH_LOG.md` | every run, in order, with diagnostics |
| `DECISIONS.md` | design decisions and the evidence that forced each |
| `MODELS.md` | pretrained checkpoint, revision, license, adaptation |
| `artifacts/EXPERIMENTS.md` | machine-generated experiment registry |

## Hardware used

NVIDIA RTX 5880 Ada (49 GB), CUDA 12.8, torch 2.11, transformers 5.5.4.
A full 30-epoch run takes ~10 minutes.

> **Environment note.** `transformers>=5` honours the dtype stored in a
> checkpoint. `microsoft/deberta-v3-large` ships **fp16** weights, and training
> fp16 master weights with AdamW produces NaN on the first optimizer step. The
> encoder is therefore loaded with `dtype=torch.float32`; mixed precision comes
> from autocast only.
