# SciEvent TARS-SciEvent: End-to-End Method-First AI Agent Execution Playbook

**Purpose.** This is the operational source of truth for an AI coding/research agent whose job is to set up the official SciEvent benchmark/evaluator correctly and spend the compute budget on **our method**, TARS-SciEvent. It explicitly does **not** require reproducing OneIE, DEGREE, EEQA, or other published baselines.

**Primary training checkpoint.** `answerdotai/ModernBERT-large`, full fine-tuning first, bf16, single GPU, measured peak train **and** inference VRAM <=30 GiB. Alternative backbones are optional controlled ablations only after the method pipeline is stable.

**Historical comparison policy.** Published SciEvent numbers may be copied from the paper into result tables as **paper-reported references**, with exact provenance. They are not rerun in this plan. Never label a paper-reported number as “reproduced”.

**Official source pin.** Use the official repository `desdai/SciEvent`, branch `EMNLP-2025`, pinned at commit `49cf9769c28f5ed8b04f61440b96163f50bca6f2`. Record the commit and SHA256 hashes of generated split/evaluator artifacts in every final experiment package.

---

## 0. Non-negotiable agent rules

1. **DO NOT TRAIN OR REPRODUCE ONEIE, DEGREE, OR EEQA.** Do not create their legacy Conda environments and do not download their model checkpoints merely to match paper numbers.
2. Baseline directories may be read or invoked only when they contain an **official protocol dependency** such as `baselines/ONEIE/EM_overlap_eval.py` or a data-format converter. Using an evaluator file is not baseline reproduction.
3. Spend the experiment budget on TARS-H0/H1/H2/H3/H4/H5, diagnostics, ablations, multi-seed confirmation, and error analysis.
4. Never tune architecture, learning rate, loss weights, thresholds, decoding, context length, or prompts on test. Test may be read only after a frozen-final marker is written from train/dev decisions.
5. Use one canonical headline protocol: the official SciEvent released annotations + released preprocessing/splitting code + official evaluator. Any alternative split is out of the main path unless explicitly requested later.
6. A SciEvent window can contain multiple events. Never flatten the dataset into “one inference sample per gold event” in a way that leaks gold event count or identity. The main model predicts an event **set** at the window level.
7. The official evaluator is the headline metric implementation. Internal metrics are debugging only. Final dev/test predictions must be exported into the exact official schema and scored by the untouched upstream evaluator.
8. Before scoring, assert prediction ID set == gold ID set. Never benefit from evaluator behavior that ignores unknown/missing IDs.
9. Treat official span IoU matching as `intersection / union > 0.5`, not `>= 0.5`. Preserve the upstream one-to-one matching behavior.
10. Keep Agent, PrimaryObject, SecondaryObject separate from the nine semantic roles. They are trigger-structure components and are excluded from semantic Arg-I/Arg-C in the released evaluator.
11. Never silently continue with missing CSCW abstracts, empty reconstructed windows, broken span offsets, or truncated gold spans.
12. All final training and inference runs must measure actual peak VRAM. Hard gate: <=30 GiB observed; operational target: <=28 GiB.
13. All reported runs must record upstream commit, overlay commit, split hashes, evaluator hash, checkpoint revision SHA, seed, config, environment, metrics, VRAM, runtime, and checkpoint SHA256.
14. Published baseline numbers are external reference points. Statistical significance claims may be made against our own matched H0/component ablations when predictions are available, **not** against paper-only numbers unless prior predictions are also available.
15. Never guarantee SOTA. If exact split identity with the paper is not verified, say “official-code regenerated split” and do not imply perfect apples-to-apples equivalence.

---

## 1. End-state repository layout

Create a clean overlay. The official repository remains read-only by convention.

```text
workspace/
├── vendor/
│   └── SciEvent/                     # exact pinned official repo
├── private_data/
│   └── cscw/                         # user-supplied, legally obtained CSCW abstracts
├── checkpoints/
│   ├── manifest.json
│   ├── preprocess-bart-tokenizer/    # tokenizer files only; official data script dependency
│   ├── modernbert-large/             # PRIMARY trainable checkpoint
│   └── optional_backbones/           # only after TARS is stable
├── data/
│   ├── official/
│   │   ├── all_data.json
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   │   ├── train.oneie.json
│   │   ├── dev.oneie.json
│   │   └── test.oneie.json
│   └── manifests/
│       ├── official_files.sha256
│       ├── split_audit.json
│       └── protocol_manifest.json
├── src/
│   └── scievent_tars/
│       ├── data/
│       │   ├── schema.py
│       │   ├── reader.py
│       │   ├── tokenizer_map.py
│       │   └── collator.py
│       ├── modeling/
│       │   ├── encoder.py
│       │   ├── discourse.py
│       │   ├── event_set.py
│       │   ├── span_candidates.py
│       │   ├── tuple_query.py
│       │   ├── prototypes.py
│       │   ├── argument_set.py
│       │   └── model.py
│       ├── training/
│       │   ├── losses.py
│       │   ├── trainer.py
│       │   ├── scheduler.py
│       │   └── registry.py
│       └── evaluation/
│           ├── export_official.py
│           ├── official_wrapper.py
│           ├── breakdowns.py
│           ├── oracles.py
│           └── bootstrap.py
├── configs/
│   └── tars/
│       ├── h0_span.yaml
│       ├── h1_tuple.yaml
│       ├── h2_proto.yaml
│       ├── h3_adaptive_proto.yaml
│       ├── h4_rhetoric.yaml
│       └── role_definitions.yaml
├── scripts/
│   ├── 00_preflight.sh
│   ├── download_checkpoints.py
│   ├── setup_official_scievent.py
│   ├── audit_split.py
│   ├── verify_official_protocol.py
│   ├── train_tars.py
│   ├── predict_tars.py
│   ├── run_oracles.py
│   ├── freeze_final.py
│   ├── measure_vram.py
│   └── final_report.py
├── tests/
│   ├── test_metric_contract.py
│   ├── test_span_roundtrip.py
│   ├── test_multi_event_window.py
│   ├── test_candidate_recall.py
│   └── test_official_export.py
├── artifacts/
│   ├── audits/
│   ├── runs/
│   ├── predictions/
│   ├── metrics/
│   └── final/
├── requirements-tars.txt
└── AGENTS.md
```

There is intentionally no `reproduce_oneie.sh`, no baseline config overlay, and no baseline training checkpoint directory.

---

## 2. Phase 0 — bootstrap, machine audit, and exact upstream pin

### 2.1 Machine assumptions

Target: Linux x86_64, one NVIDIA GPU. The physical card may have more than 30 GiB, but our run is invalid if measured peak allocation exceeds 30 GiB.

Record:

```bash
mkdir -p artifacts/audits
{
  date -u
  uname -a
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  python --version || true
  conda --version || true
  git --version
} | tee artifacts/audits/hardware.txt
```

### 2.2 Clone and pin official SciEvent

```bash
mkdir -p vendor
git clone --branch EMNLP-2025 https://github.com/desdai/SciEvent.git vendor/SciEvent
git -C vendor/SciEvent checkout 49cf9769c28f5ed8b04f61440b96163f50bca6f2

ACTUAL=$(git -C vendor/SciEvent rev-parse HEAD)
test "$ACTUAL" = "49cf9769c28f5ed8b04f61440b96163f50bca6f2"
test -z "$(git -C vendor/SciEvent status --porcelain)"
printf '%s\n' "$ACTUAL" > artifacts/audits/upstream_commit.txt
```

Never use only a moving branch name in a paper/release command.

### 2.3 Hash the official evaluator immediately

The headline extraction evaluator is:

```text
vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py
```

Hash it without modifying it:

```bash
sha256sum vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py   > artifacts/audits/official_evaluator.sha256
```

This baseline-directory path is used because the benchmark authors placed the evaluator there. We are not using the OneIE model.

---

## 3. Phase 1 — one modern environment, no baseline environments

Create only the TARS environment:

```bash
conda create -n scievent-tars python=3.11 -y
conda activate scievent-tars

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-tars.txt

python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

python -m pip freeze > artifacts/audits/pip_freeze.initial.txt
```

If the host driver does not support that CUDA wheel, install the nearest supported official PyTorch wheel and freeze the exact resulting environment. The scientific requirement is the recorded environment + VRAM gate, not a particular CUDA minor version.

Do **not** run:

```text
conda env create -f oneie.yml
conda env create -f degree.yml
conda env create -f eeqa.yml
```

Those environments are unnecessary for this method-first workflow.

---

## 4. Phase 2 — checkpoint acquisition: only what our method/protocol needs

### 4.1 Primary model checkpoint

Primary:

```text
answerdotai/ModernBERT-large
```

Download by resolving `main` to an immutable Hugging Face revision SHA, then train from that exact local snapshot.

### 4.2 Preprocessing tokenizer dependency

Official `prepare_segmentation.py` defaults to the `facebook/bart-large` tokenizer. We need tokenizer assets for protocol-compatible preprocessing, **not** the BART model for training.

Store it under:

```text
checkpoints/preprocess-bart-tokenizer/
```

Download only tokenizer/config files when possible.

### 4.3 Optional backbone ablations

Only after TARS-H0/H1/H2/H3 are stable, optionally test matched-head controls such as:

```text
microsoft/deberta-v3-large
allenai/scibert_scivocab_cased
```

These are method/backbone ablations, not baseline reproductions. Do not spend early compute on them.

Run:

```bash
python scripts/download_checkpoints.py   --profile primary   --output checkpoints   --manifest checkpoints/manifest.json

python scripts/download_checkpoints.py   --profile preprocess   --output checkpoints   --manifest checkpoints/manifest.json
```

The checkpoint manifest must contain at minimum:

```json
{
  "modernbert-large": {
    "repo_id": "answerdotai/ModernBERT-large",
    "revision_sha": "<resolved sha>",
    "local_dir": "checkpoints/modernbert-large"
  },
  "preprocess-bart-tokenizer": {
    "repo_id": "facebook/bart-large",
    "revision_sha": "<resolved sha>",
    "local_dir": "checkpoints/preprocess-bart-tokenizer",
    "tokenizer_only": true
  }
}
```

Never leave `revision=main` as the only provenance.

---

## 5. Phase 3 — set up SciEvent official data completely

This phase exists only to make the benchmark protocol executable. Do not train published models.

### 5.1 CSCW raw-text gate

The official repository does not distribute CSCW raw abstracts because of licensing. Place legally obtained files in:

```text
private_data/cscw/
```

Then copy/symlink them into:

```text
vendor/SciEvent/SciEvent_data/abstracts_texts/
```

Before preprocessing, enumerate every `doc_id` referenced by the annotation/event-segmentation files and require a matching raw abstract file. Write missing IDs to:

```text
artifacts/audits/missing_abstracts.txt
```

Hard gate:

```text
missing_abstract_count == 0
```

If not, stop. `prepare_segmentation.py` can otherwise emit empty windows; never train on those silently.

### 5.2 Run official preprocessing scripts directly in the TARS environment

Do not create DEGREE's environment. We need the **official scripts**, not the DEGREE model.

Use an explicit temporary companion file and the locally pinned BART tokenizer:

```bash
conda activate scievent-tars
cd vendor/SciEvent

mkdir -p SciEvent_data/DEGREE/processed
TMP_TEXTS=$(mktemp -t scievent_texts_only.XXXXXX.jsonl)

python data_scripts/shared/prepare_segmentation.py   --annotation SciEvent_data/annotated/event_extraction_finetune_model.jsonl   --event_seg SciEvent_data/annotated/event_seg.jsonl   --abstract_dir SciEvent_data/abstracts_texts   --output "$TMP_TEXTS"   --hf_model ../../checkpoints/preprocess-bart-tokenizer   --hf_cache ../../checkpoints/hf-cache

python data_scripts/shared/prepare_all_data.py   --annotation SciEvent_data/annotated/event_extraction_finetune_model.jsonl   --texts "$TMP_TEXTS"   --output SciEvent_data/DEGREE/processed/all_data.json

rm -f "$TMP_TEXTS"
```

This preserves official reconstruction logic while avoiding a legacy training environment.

### 5.3 Generate the official-code split once and freeze it

Upstream `split_data.py` uses seed 42 but iterates over Python sets. To make our generated artifact reproducible, freeze `PYTHONHASHSEED` and record it:

```bash
export PYTHONHASHSEED=0
python data_scripts/shared/split_data.py
```

This is the **official-code regenerated split**. Its file hashes, not an assumed paper split identity, are the reproducibility contract for our run.

If the authors later provide exact paper split files/hashes, prefer those and record that exact provenance. Until then, do not claim that a regenerated split is byte-identical to the unpublished paper split.

### 5.4 Create evaluator-compatible gold files only

The official OneIE evaluator expects `sent_id`. Use the official data converter only:

```bash
bash data_scripts/ONEIE/wnd_id_rename.sh
```

This simply converts `wnd_id -> sent_id`; it does not train OneIE.

Do not run the EEQA converter unless another analysis explicitly needs its schema.

### 5.5 Freeze generated data into our overlay

Copy the generated artifacts out of the vendor tree so later accidental preprocessing cannot mutate a running experiment:

```bash
cd ../..

mkdir -p data/official data/manifests

cp vendor/SciEvent/SciEvent_data/DEGREE/processed/all_data.json data/official/
cp vendor/SciEvent/SciEvent_data/DEGREE/all_splits/train.json data/official/
cp vendor/SciEvent/SciEvent_data/DEGREE/all_splits/dev.json data/official/
cp vendor/SciEvent/SciEvent_data/DEGREE/all_splits/test.json data/official/
cp vendor/SciEvent/SciEvent_data/ONEIE/all_splits/train.oneie.json data/official/
cp vendor/SciEvent/SciEvent_data/ONEIE/all_splits/dev.oneie.json data/official/
cp vendor/SciEvent/SciEvent_data/ONEIE/all_splits/test.oneie.json data/official/

sha256sum data/official/* > data/manifests/official_files.sha256
```

Then make the training code read only from `data/official/`.

### 5.6 Mandatory data sanity checks

The setup script must fail if any is true:

```text
any split is empty
any expected wnd_id/sent_id is missing
any reconstructed sentence is empty
any span has start < 0
any span has end <= start
any span end exceeds number of words
any event type is outside {Background, Method, Result, Conclusion}
any semantic role is outside the official schema
raw abstract coverage <100%
generated split files change after rerunning with same hash seed/environment
```

Also summarize, for TRAIN only when choosing capacities:

```text
max events/window
event count distribution
argument span-width distribution
same-role multiplicity/event
subword length distribution under ModernBERT
role frequencies
event-type frequencies
```

Never use dev/test maxima to choose architectural capacities.

---

## 6. Phase 4 — audit the generated official split, but do not fork the main protocol

Run:

```bash
python scripts/audit_split.py   --train data/official/train.json   --dev data/official/dev.json   --test data/official/test.json   --out data/manifests/split_audit.json
```

Record intersections for:

```text
doc_id
wnd_id
normalized text SHA256
```

This audit is disclosure/diagnosis, not a reason to silently create a new evaluation protocol. The main result remains on the frozen official-code split generated above.

If overlaps are found, report them in the paper/reproducibility appendix. Do not call them “leakage” without checking the actual generated files and task semantics.

---

## 7. Phase 5 — lock the official evaluator contract before model work

Use the untouched:

```text
vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py
```

Verified behavior to preserve:

```text
IoU span match: intersection / union > 0.5
Arg-I: same event type + matching argument span; role ignored
Arg-C: same event type + matching argument span + same role
Agent / PrimaryObject / SecondaryObject excluded from semantic argument score
argument score is trigger-span insensitive
matching is greedy one-to-one inside each sent_id/window
precision / recall / F1 are micro aggregated
```

### 7.1 Synthetic evaluator tests

Before training TARS, make `pytest -q tests/test_metric_contract.py` cover:

```text
identical spans -> match
IoU exactly 0.5 -> no match
IoU 0.500...+ -> match
duplicate predictions against one gold -> one TP plus FP(s)
one prediction against duplicate gold -> at most one TP
correct span + wrong semantic role -> Arg-I hit, Arg-C miss
correct role + partial span >0.5 -> IoU hit, EM miss
empty prediction list -> defined zero behavior
empty gold list -> no crash
unknown sent_id -> document upstream behavior
```

### 7.2 Strict wrapper around the official evaluator

Our wrapper must reject malformed predictions **before** calling upstream:

```text
prediction sent_id set == gold sent_id set
one JSON object per gold window
all word spans valid
only official event types
only official roles
no NaN/inf
```

Then subprocess the upstream evaluator and save raw stdout + parsed JSON.

Headline command:

```bash
python -m scievent_tars.evaluation.official_wrapper   --official-evaluator vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py   --gold data/official/dev.oneie.json   --pred artifacts/predictions/<run>/dev.oneie.json   --require-exact-id-set   --raw-stdout artifacts/metrics/<run>/dev.official.txt   --write-json artifacts/metrics/<run>/dev.official.json
```

Do not reimplement headline scoring and then merely assert it is “equivalent”.

### 7.3 Paper-reported references

Maintain a separate machine-readable file such as:

```text
artifacts/reference/paper_reported_results.json
```

Every row must contain:

```text
paper title
table/section
model name
metric
reported value
protocol description
source URL/citation
reproduced = false
```

These values are comparison references only. They never gate whether our environment is “correct”; evaluator tests and data hashes do.

---

## 8. Phase 6 — canonical TARS data model

### 9.1 Do not create one inference sample per gold event

A window can contain multiple event mentions and multiple event types. Flattening gold events into separate inference samples leaks gold event count/identity. TARS must operate at the **window level** and predict a set of events.

Canonical window object:

```python
WindowExample(
    doc_id: str,
    wnd_id: str,
    words: list[str],
    sentence_starts: list[int],
    events: list[GoldEvent],
    domain: str | None,
)

GoldEvent(
    event_id: str,
    event_type: Literal["Background", "Method", "Result", "Conclusion"],
    action_span: Span,                 # released event["trigger"]
    agent_spans: list[Span],           # role == Agent
    primary_object_spans: list[Span],  # role == PrimaryObject
    secondary_object_spans: list[Span],# role == SecondaryObject
    arguments: list[GoldArgument],     # the 9 semantic roles only
)

GoldArgument(
    role: Literal[
      "Context","Purpose","Method","Result","Analysis","Challenge",
      "Ethical","Implication","Contradiction"
    ],
    span: Span,
)
```

All spans use **word-token, end-exclusive coordinates**, exactly matching the official representation. Never train/output subword offsets.

### 9.2 Word/subword mapping

Call the fast tokenizer with:

```python
tok = tokenizer(
    words,
    is_split_into_words=True,
    truncation=False,
    add_special_tokens=True,
)
word_ids = tok.word_ids()
```

Build `word -> [first_piece, last_piece_exclusive]`. Model span heads score word-level states obtained by pooling each word's pieces. Round-trip unit tests must show that every gold word span exports exactly to its original indices.

Never silently truncate a gold span. Before training, compute subword length distribution. If any example exceeds configured `max_length`, either increase length or implement an explicitly reported chunk strategy that preserves all gold spans. ModernBERT supports long context, but longer context is an experiment, not an assumption.

---

## 9. TARS-SciEvent architecture: nested event set -> trigger tuple -> argument span sets

Working name: **TARS-SciEvent — Tuple-Anchored Rhetorical Span-set Extraction**.

The central causal hypothesis is that SciEvent's semantic arguments are easier to identify/classify when the model represents an event compositionally through its Agent–Action–Object trigger structure, while rare discourse roles benefit from annotation-definition priors. The architecture must predict a set of events at the window level so it does not receive gold event identity.

### 10.1 Encoder

Primary checkpoint: `answerdotai/ModernBERT-large` (395M). Keep this fixed through the core method funnel. Optional matched-head backbone ablations (`microsoft/deberta-v3-large`, `allenai/scibert_scivocab_cased`) come only after the architecture is stable.

For word states:

```text
piece states = Encoder(input_ids, attention_mask)
word state h_i = mean(piece states belonging to word i)
```

Use bf16 when supported. Full fine-tuning is the primary regime.

### 10.2 Lightweight rhetorical hierarchy (optional component, not assumed helpful)

Pool word states per sentence:

```text
s_m = mean(h_i for words in sentence m)
S' = 2-layer TransformerEncoder(S)
h'_i = LayerNorm(h_i + W_s * S'[sentence(i)])
```

Start with this component OFF in the matched span baseline. Add it only after tuple/prototype diagnostics. Do not make a long-context claim from this component.

### 10.3 Event-set decoder

Measure the maximum number of gold events per training window. Set:

```text
K_event = max_train_events_per_window + 1
```

Cap only if an outlier makes the value unreasonable, and document the cap. Never derive `K_event` from dev/test.

Create `K_event` learned event queries and use a small Transformer decoder (initially 2 layers, hidden projected to encoder hidden size) attending to token/word states. Each slot predicts:

```text
p(event_type in {Background, Method, Result, Conclusion, NONE})
Action start/end span
Agent span or NULL
PrimaryObject span or NULL
SecondaryObject span or NULL
```

If training data shows multiple Agent/PrimaryObject instances for one event, replace the single component slot with a tiny set head for that component. This decision must be based on train statistics only.

### 10.4 Event Hungarian matching

Match predicted event slots to gold events with a permutation-invariant Hungarian assignment. Initial cost:

```text
C_event(pred k, gold j) =
    2.0 * CE(event_type_k, gold_type_j)
  + 4.0 * span_cost(Action_k, Action_j)
  + 0.5 * span_cost(Agent_k, Agent_j)
  + 0.5 * span_cost(PrimaryObject_k, PrimaryObject_j)
  + 0.25* span_cost(SecondaryObject_k, SecondaryObject_j)
```

`span_cost` should combine start CE, end CE, and optionally 1-IoU at word level. Loss weights are starting values only; tune on dev in a small predefined search. Unmatched event slots train toward `NONE`.

This event-set layer is mandatory for official-comparable TARS. A per-gold-event model may be used only as an ORACLE diagnostic.

### 10.5 Differentiable trigger tuple query

For Action/Agent/PrimaryObject/SecondaryObject, compute a soft span/component representation from the predicted start/end distributions instead of feeding a hard gold span into the main forward path:

```text
g_start(c) = Σ_i p_start_c(i) h'_i
g_end(c)   = Σ_i p_end_c(i) h'_i
g_c        = MLP([g_start(c); g_end(c); presence_prob(c)])
```

For nullable components, include a learned NULL vector weighted by null probability.

Use expected event-type embedding:

```text
e_type = Σ_t p(type=t) E_type[t]
```

Construct tuple query:

```text
q_tuple = LayerNorm(W_t [q_event ; g_Agent ; g_Action ; g_PO ; g_SO ; e_type])
```

This avoids gold-trigger teacher forcing in the primary model and makes train/inference conditioning consistent. Still run a gold-trigger oracle separately to quantify the ceiling.

### 10.6 Global candidate span proposal

Enumerating all O(n^2) spans is unnecessary. On TRAIN ONLY, compute the semantic argument span-width distribution. Choose:

```text
max_span_width = min(64, ceil(train_span_width_p99_5))
```

with a minimum reasonable floor such as 8 or 16 if the percentile is very small. Record the selected value.

Boundary proposal heads score start/end words independently. At inference, take top-K starts and ends, form valid spans where `end > start` and width <= max_span_width, then keep top `K_span` by boundary score. Initial diagnostics:

```text
K_start = 64
K_end   = 64
K_span  = 512
```

At training time, inject all gold argument spans into the candidate set even if the proposal head misses them. Track `candidate_recall`; require >=99% on train and preferably >=98% on dev before attributing errors to the role decoder.

Candidate representation:

```text
z_(i,j) = MLP([
    h'_i ; h'_(j-1) ; attentive_pool(h'_i...h'_(j-1)) ;
    width_embedding(j-i) ; relative_sentence_distance_to_action
])
```

### 10.7 Frequency-adaptive codebook prototypes

Use the exact role definitions from the SciEvent annotation codebook/appendix. Do not invent descriptions. Save them in `configs/tars/role_definitions.yaml` with a source/page note and checksum.

Before fine-tuning, encode each role definition with the same frozen initial backbone and save a fixed definition vector `p_def[r]`. Add a learned empirical role vector `p_emp[r]`.

Let `n_r` be TRAIN-ONLY role frequency and choose a global `kappa` on dev from a small grid, e.g. {5, 20, 50}:

```text
lambda_r = kappa / (kappa + n_r)
p_role[r] = lambda_r * W_def p_def[r] + (1-lambda_r) * p_emp[r]
```

Rare roles remain closer to their semantic definition; frequent roles learn more freely from data. Do not tune a separate lambda for each rare role.

Optional anchor regularization:

```text
L_proto = Σ_r lambda_r * ||normalize(p_emp[r]) - normalize(W_def p_def[r])||^2
```

Ablate both the definition initialization and frequency-adaptive blending.

### 10.8 Per-event, per-role argument set decoder

For each predicted event slot `k`, each semantic role `r`, and each role slot `m`, create:

```text
q_arg[k,r,m] = MLP([q_tuple[k] ; p_role[r] ; E_slot[m]])
```

Set initial `K_arg_role` from the TRAIN maximum number of same-role arguments per event, plus one NULL-capacity slot. If this max is noisy, use train p99 plus one and record dropped-capacity risk. Never inspect test to pick this.

Score each role slot against every candidate span plus a NULL candidate:

```text
score(k,r,m,a) = q_arg[k,r,m]^T W_arg z_a + b
```

For each matched gold event and each role independently, Hungarian-match role slots to the gold span set for that role. Remaining slots map to NULL. This gives permutation-invariant support for multiple arguments of one role.

At inference:

```text
1. discard event slots predicted NONE
2. decode event type and trigger components
3. for each role slot choose best span or NULL
4. remove exact duplicate spans within the same event+role
5. do not globally suppress a span merely because another role uses it
6. export word-level end-exclusive offsets
```

Tune only a single global NULL-margin/calibration value on dev if calibration is necessary. Avoid per-role thresholds for roles with tiny dev counts.

### 10.9 Core loss

For matched events:

```text
L = L_event_type
  + λ_action * L_action_span
  + λ_tuple  * (L_agent + L_primary_object + L_secondary_object)
  + λ_arg    * L_argument_set
  + λ_bnd    * L_boundary_proposal
  + λ_proto  * L_prototype_anchor
```

Starting loss weights for a first dev run:

```text
λ_action = 1.0
λ_tuple  = 1.0
λ_arg    = 2.0
λ_bnd    = 0.5
λ_proto  = 0.05
```

These are hypotheses, not facts. Tune only on dev and only after the matched baseline is stable.

### 10.10 Why this is not just PAIE + a new encoder

The novelty to test is the **nested window-level event-set model plus compositional Agent–Action–Object tuple anchoring plus frequency-adaptive annotation-definition role prototypes**. Span-set extraction and Hungarian matching alone are prior art and must not be claimed as novel. The novelty claim survives only if matched-backbone ablations show gains from tuple anchoring and prototype regularization.

---

## 10. Phase 7 — TARS implementation order

Implement in this exact order so bugs are localized.

### 11.1 Data and round-trip first

Implement `schema.py`, `reader.py`, `tokenizer_map.py`, and tests. Acceptance:

```text
100% gold spans round-trip word -> subword pooling -> exported word coordinates
all event mentions preserved
all 9 semantic roles preserved
Agent/PO/SO separated from semantic args
no example silently truncated
```

### 11.2 Matched encoder + simple event/span baseline

Before prototypes/rhetoric, build a minimal fair baseline on the same backbone:

```text
encoder
+ event-set slots
+ type/action/Agent/PO/SO heads
+ candidate spans
+ independent semantic role classification or simple role slots
```

This is TARS-H0. It answers whether a modern span formulation itself is enough.

### 11.3 TARS-H1: tuple anchoring

Replace generic event-slot conditioning of arguments with `q_tuple`. Everything else stays fixed. Run at least seeds 13 and 42 on dev.

Falsification: if H1 does not improve dev Arg-C IoU and/or EM consistently over H0, tuple anchoring is not supported. Do not hide the negative result.

### 11.4 TARS-H2: definition prototype bank

Add fixed codebook definition vectors but no frequency adaptation. Compare H2 vs H1.

### 11.5 TARS-H3: frequency-adaptive prototype blend

Add `lambda_r = kappa/(kappa+n_r)`. Evaluate overall Arg-C plus role-wise frequent/rare results. Reject if rare-role gains come with a material overall drop.

### 11.6 TARS-H4: rhetorical hierarchy

Only after H1-H3. Compare OFF vs 2-layer sentence hierarchy, keeping encoder/head fixed. If no stable gain, leave it out of the final method even though it appears in the initial research hypothesis.

### 11.7 TARS-H5: domain-robust training, optional

Do not feed explicit domain ID to the final inference model by default. First try domain-balanced batches. If domain imbalance remains severe, test GroupDRO/worst-group weighting as a separate component. Report it separately if it changes training assumptions.

---

## 11. Phase 8 — training recipe under <=30 GiB

### 12.1 Primary full-finetuning recipe

Start with:

```yaml
backbone: answerdotai/ModernBERT-large
precision: bf16
full_finetuning: true
max_length: 512
micro_batch_size: 1
grad_accumulation: 16
gradient_checkpointing: true
optimizer: adamw
backbone_lr: 1.0e-5
head_lr: 3.0e-4
weight_decay: 0.01
warmup_ratio: 0.10
max_epochs: 30
early_stop_metric: dev/arg_c_iou_f1
early_stop_patience: 5
clip_grad_norm: 1.0
seed: 42
```

Use length bucketing. Evaluate once per epoch. Save best checkpoint by **dev Arg-C IoU F1**, with EM as a secondary diagnostic.

### 12.2 Cheap LR search

Do not brute-force. First run one seed on a reduced epoch budget:

```text
backbone_lr in {5e-6, 1e-5, 2e-5}
head_lr     in {1e-4, 3e-4}
```

Keep the rest fixed. Promote only the best one or two configurations to two-seed confirmation.

### 12.3 VRAM fallback ladder

If observed peak exceeds 28 GiB, reduce memory in this order while keeping full fine-tuning:

```text
1. confirm micro-batch = 1
2. enable/verify gradient checkpointing
3. dynamic length bucketing and padding-to-batch-max
4. PyTorch SDPA / supported efficient attention
5. reduce max_length only if audit proves no gold truncation
6. switch AdamW optimizer states to 8-bit; label this optimizer regime explicitly
```

Do not jump directly to LoRA and then compare it as “the same” full-FT regime. A PEFT branch is allowed but reported separately.

### 12.4 Internal VRAM measurement

At the beginning of training/inference:

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

At the end:

```python
peak_alloc_gib = torch.cuda.max_memory_allocated() / 2**30
peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
```

Write both to the run manifest. Also run external `nvidia-smi` polling. Hard assertion for final runs:

```python
assert peak_alloc_gib <= 30.0
```

Operational preference:

```text
peak_alloc_gib <= 28.0
```

Measure inference separately.

---

## 12. Phase 9 — diagnostic experiments before final combination

Run these on train/dev only.

```text
O1 normal predicted type + predicted trigger tuple + predicted arguments
O2 GOLD semantic argument spans -> classify role only
O3 GOLD event type -> predicted trigger/arguments
O4 GOLD trigger tuple -> predicted semantic arguments
O5 GOLD type + GOLD tuple -> predicted semantic arguments
```

O2 directly estimates the role-classification ceiling with perfect boundaries. O4 estimates how much argument performance is lost through trigger conditioning. These are ORACLE results and never enter the SOTA table.

Optional backbone ablations using the same TARS-H0 head, only after core method stabilization:

```text
B1 SciBERT cased
B2 DeBERTa-v3-large
B3 ModernBERT-large
```

Do not run these early. If run later, keep head/data/schedule/dev protocol matched and treat them as optional backbone ablations.

Context diagnostic:

```text
C1 max_length 256
C2 max_length 512
C3 max_length 1024 only if length audit justifies it
```

Do not spend compute on 8k context unless the data demonstrates a real truncation/long-distance bottleneck.

---

## 13. Phase 10 — official export format

TARS predicts at window level. Group all non-NONE event slots by `wnd_id` and export one official entry per window. The exported event uses:

```text
event_type = predicted event type
trigger    = predicted Action span
arguments:
  Agent
  PrimaryObject
  SecondaryObject
  plus semantic argument roles
```

The exact object keys/order should be copied from the generated OneIE/DEGREE gold schema. Do not invent a new evaluator input format.

Before scoring:

```text
set(pred sent_id/wnd_id) == set(gold sent_id/wnd_id)
all spans satisfy 0 <= start < end <= number_of_words
all event types are in the official four-type schema
all roles are in official schema
no NaN scores
no duplicate event IDs
```

For windows with no predicted event, export an entry with an empty `event_mentions` list rather than omitting the ID.

---

## 14. Phase 11 — error analysis and decision rules

Every promoted experiment must write breakdowns by:

```text
domain
event type
role
argument span width
window/document length
action-to-argument sentence distance
single-sentence vs cross-sentence argument
same-role multiplicity
candidate proposal recall
seen/unseen train terminology proxy
```

Also record boundary-only errors by comparing Arg-I EM vs IoU. A gain only in IoU with no EM gain may indicate looser boundaries; do not overstate it as better exact extraction.

Decision rules:

```text
Promote a component if it improves dev Arg-C IoU in >=2 seeds and does not create a clear EM collapse.
Prefer a smaller reliable gain over a larger unstable single-seed gain.
Do not combine two components just because each once had a positive run.
If a component helps only one domain, report that and test robustness before inclusion.
If candidate span recall <98% dev, fix proposal recall before changing role modeling.
```

---

## 15. Phase 12 — method-first experiment funnel and tuning budget

The training budget must primarily answer whether our method components work, not whether old baselines can be recreated.

### 15.1 Stage A — engineering smoke test

Use a tiny TRAIN subset only. The goal is no crash, correct shapes, overfit capability, exact export round-trip, and VRAM logging.

Acceptance:

```text
can overfit 16-32 windows
loss decreases
event slot permutation does not break labels
gold spans survive tokenizer mapping
prediction exporter passes schema validation
official evaluator runs end-to-end on dev-format predictions
```

Never use smoke-test scores in the paper.

### 15.2 Stage B — TARS-H0 matched internal control

H0:

```text
ModernBERT-large
+ window-level event-set decoder
+ Action / Agent / PrimaryObject / SecondaryObject heads
+ candidate semantic spans
+ simple learned role representations
NO tuple anchoring
NO definition prototypes
NO rhetorical hierarchy
```

This is the most important internal baseline because we own its predictions and can do paired ablations/statistics without reproducing external models.

Run seed 42 first. If implementation is stable, confirm with seed 13 before promoting more complexity.

### 15.3 Stage C — causal component funnel

Run in order:

```text
H1 = H0 + compositional tuple query
H2 = H1 + fixed codebook-definition prototypes
H3 = H2 + frequency-adaptive prototype blending
H4 = H3 + rhetorical hierarchy, only if diagnostics motivate it
H5 = best prior + domain-robust batching/GroupDRO, optional
```

A component is promoted only when:

```text
dev Arg-C IoU improves in >=2 seeds
no material Arg-C EM collapse
gain is not only one high-support role/domain artifact
VRAM remains <=30 GiB
```

If H1 is neutral/negative, do not keep it just because it is part of the original idea. The final method is the set of empirically supported components.

### 15.4 Stage D — small hyperparameter search

For the strongest architecture, run a compact train/dev search only.

Recommended grid:

```text
backbone_lr: {5e-6, 1e-5, 2e-5}
head_lr: {1e-4, 3e-4}
kappa: {5, 20, 50}             # only after prototype component is supported
K_span: {256, 512}              # only if candidate recall allows
max_length: {512, 1024}         # only if length audit shows a reason
```

Do not Cartesian-product all values. Use sequential diagnosis:

```text
1. choose LR pair
2. freeze LR
3. choose kappa if needed
4. tune span proposal capacity only if recall/VRAM requires
5. test longer context only if data says 512 is insufficient
```

### 15.5 Stage E — freeze final before test

A final config may be frozen only from TRAIN/DEV evidence:

```bash
python scripts/freeze_final.py \
  --config configs/tars/<winner>.yaml \
  --out artifacts/final/frozen_config.yaml

sha256sum artifacts/final/frozen_config.yaml \
  > artifacts/final/frozen_config.sha256

date -u > artifacts/final/FROZEN_BEFORE_TEST.txt
```

After this marker exists, no hyperparameter/architecture changes may be justified using test.

---

## 16. Phase 13 — final seeds, statistics, and comparison policy

Recommended final seeds:

```text
[13, 42, 101]
```

Prefer five if compute permits:

```text
[13, 21, 42, 87, 101]
```

Required rows:

```text
Paper-reported best comparable tuning model  # external reference, reproduced=false
TARS-H0                                      # our matched internal control
TARS-H1
TARS-H2
TARS-H3
TARS-H4/H5 only if promoted
TARS-FINAL
```

For our runs report:

```text
mean
standard deviation
median
all individual seeds
peak train VRAM
peak inference VRAM
runtime
```

Do not select the best test seed as the headline number.

### 16.1 Statistics we can honestly compute without reproducing old baselines

Because we have H0 and TARS-FINAL predictions on the same examples, run document-block paired bootstrap/randomization for:

```text
TARS-FINAL vs TARS-H0
component-on vs component-off ablations
```

For paper-only baseline numbers, report:

```text
paper reported point estimate
our mean/best point estimate
absolute margin
protocol/split comparability note
```

Do **not** write “statistically significantly better than OneIE/DEGREE/EEQA” without their predictions or a valid uncertainty estimate.

---

## 17. Phase 14 — run manifest contract

Every run writes `artifacts/runs/<experiment_id>/manifest.json`:

```json
{
  "experiment_id": "...",
  "timestamp_utc": "...",
  "upstream_scievent_commit": "49cf9769c28f5ed8b04f61440b96163f50bca6f2",
  "overlay_git_commit": "...",
  "data_protocol": "official_code_regenerated",
  "pythonhashseed": "0",
  "split_sha256": "...",
  "evaluator_sha256": "...",
  "backbone_repo_id": "answerdotai/ModernBERT-large",
  "backbone_revision_sha": "...",
  "training_regime": "full_ft",
  "external_supervised_data": false,
  "external_unlabeled_data": false,
  "seed": 42,
  "precision": "bf16",
  "max_length": 512,
  "micro_batch_size": 1,
  "grad_accumulation": 16,
  "optimizer": "AdamW",
  "backbone_lr": 1e-5,
  "head_lr": 3e-4,
  "epochs_run": 0,
  "best_dev_epoch": 0,
  "dev_metrics": {},
  "test_metrics": {},
  "peak_train_allocated_gib": null,
  "peak_train_reserved_gib": null,
  "peak_infer_allocated_gib": null,
  "peak_infer_reserved_gib": null,
  "nvidia_smi_peak_mib": null,
  "runtime_seconds": null,
  "checkpoint_sha256": null,
  "frozen_before_test_sha256": null,
  "notes": ""
}
```

A missing value is `null` with explanation; do not silently omit provenance/resource fields.

---

## 18. Phase 15 — SOTA / paper-comparison gate without baseline reproduction

The agent may use SOTA wording only if all applicable conditions pass:

```text
[ ] Latest direct SciEvent literature was searched to the execution date.
[ ] Prior comparable best is identified from a primary source.
[ ] Our result uses the official SciEvent evaluator.
[ ] Our test was untouched before final config freeze.
[ ] Multiple final seeds were run.
[ ] Train/inference measured peak VRAM <=30 GiB.
[ ] Checkpoint, code, evaluator, split, and config are frozen/hashed.
[ ] External data/training regime are disclosed.
[ ] Our score exceeds the comparable prior point estimate under a demonstrably comparable protocol.
[ ] Split identity/protocol equivalence is verified OR the claim is explicitly restricted to our regenerated official-code split.
[ ] Gain is not solely one seed, one domain, or an evaluator artifact.
```

Important wording rules:

```text
If exact paper split identity is verified:
  "exceeds the paper-reported comparable result under the same released protocol"

If evaluator is official but split identity is not byte-level verified:
  "best result in our experiments on the regenerated official-code SciEvent split"

If current literature cannot be fully verified:
  "candidate SOTA pending current-literature verification"
```

Baseline reproduction is **not** a SOTA-gate requirement in this plan.

---

## 19. Automated stop conditions

```text
STOP_DATA_01: missing raw abstract count > 0
STOP_DATA_02: any reconstructed window is unexpectedly empty
STOP_DATA_03: rerunning official setup with same environment/hash seed changes split hashes
STOP_METRIC_01: synthetic evaluator tests disagree with upstream
STOP_METRIC_02: prediction ID set != gold ID set
STOP_MODEL_01: any gold span is truncated or cannot round-trip
STOP_MODEL_02: dev candidate span recall <98%
STOP_EVENT_01: model inference receives gold event count/type/identity
STOP_VRAM_01: peak allocated >30 GiB in train or inference
STOP_TEST_01: test was evaluated before frozen-final marker
STOP_STATS_01: final scientific claim relies on one seed only
STOP_COMPARE_01: paper-reported number is labeled as reproduced
STOP_COMPARE_02: exact split equivalence is asserted without evidence
STOP_SOTA_01: current direct SciEvent literature not refreshed before SOTA wording
```

On stop: print the code, write a diagnostic artifact, repair the root cause, then resume from the failed stage.

---

## 20. Autonomous command sequence — no baseline reproduction

From the extracted agent pack:

```bash
# 1) Bootstrap pinned repo + overlay
bash bootstrap.sh "$PWD/workspace"
cd workspace

# 2) Create ONE modern environment
conda create -n scievent-tars python=3.11 -y
conda activate scievent-tars
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-tars.txt
python -m pip freeze > artifacts/audits/pip_freeze.initial.txt

# 3) Download only primary trainable checkpoint + preprocessing tokenizer
python scripts/download_checkpoints.py --profile primary \
  --output checkpoints --manifest checkpoints/manifest.json
python scripts/download_checkpoints.py --profile preprocess \
  --output checkpoints --manifest checkpoints/manifest.json

# 4) Put legally obtained CSCW abstracts in private_data/cscw/
#    then generate/freeze official data; NO baseline training
python scripts/setup_official_scievent.py \
  --workspace "$PWD" \
  --pythonhashseed 0

# 5) Audit generated split and lock evaluator contract
python scripts/audit_split.py \
  --train data/official/train.json \
  --dev data/official/dev.json \
  --test data/official/test.json \
  --out data/manifests/split_audit.json

python scripts/verify_official_protocol.py \
  --workspace "$PWD"

pytest -q tests/test_metric_contract.py \
          tests/test_span_roundtrip.py \
          tests/test_multi_event_window.py \
          tests/test_official_export.py

# 6) Method smoke test
python scripts/train_tars.py \
  --config configs/tars/h0_span.yaml \
  --seed 42 \
  --smoke-test

# 7) Full H0 internal baseline on TRAIN/DEV
python scripts/train_tars.py --config configs/tars/h0_span.yaml --seed 42
python scripts/train_tars.py --config configs/tars/h0_span.yaml --seed 13

# 8) Causal method funnel, TRAIN/DEV only
python scripts/train_tars.py --config configs/tars/h1_tuple.yaml --seed 42
python scripts/train_tars.py --config configs/tars/h1_tuple.yaml --seed 13

python scripts/train_tars.py --config configs/tars/h2_proto.yaml --seed 42
python scripts/train_tars.py --config configs/tars/h2_proto.yaml --seed 13

python scripts/train_tars.py --config configs/tars/h3_adaptive_proto.yaml --seed 42
python scripts/train_tars.py --config configs/tars/h3_adaptive_proto.yaml --seed 13

# 9) Oracles/error analysis before adding more complexity
python scripts/run_oracles.py --config <best-dev-config>
python scripts/final_report.py --stage dev-diagnostics --runs artifacts/runs

# 10) Small hyperparameter search on best architecture
#     Use dev only, sequentially; do not brute-force a giant grid.

# 11) Freeze final config BEFORE any test evaluation
python scripts/freeze_final.py \
  --config <best-dev-config> \
  --out artifacts/final/frozen_config.yaml
sha256sum artifacts/final/frozen_config.yaml \
  > artifacts/final/frozen_config.sha256
date -u > artifacts/final/FROZEN_BEFORE_TEST.txt

# 12) Final multi-seed test runs
for s in 13 42 101; do
  python scripts/train_tars.py \
    --config artifacts/final/frozen_config.yaml \
    --seed "$s" \
    --final-test
done

# 13) Aggregate official scores, ablations, VRAM, bootstrap vs our H0, SOTA gate
python scripts/final_report.py \
  --runs artifacts/runs \
  --paper-reference artifacts/reference/paper_reported_results.json \
  --out artifacts/final/report
```

Explicitly forbidden in this workflow:

```text
training baselines/ONEIE/train.py
training baselines/DEGREE/...
running EEQA training scripts
downloading baseline checkpoints merely for reproduction
blocking TARS work because paper baseline was not reproduced
```

---

## 21. What the final contribution may say

Only keep clauses supported by ablations:

> TARS-SciEvent predicts scientific events as a latent window-level set, represents each event compositionally through Agent–Action–Object trigger structure, and uses the resulting tuple to condition permutation-invariant semantic argument span sets. To address sparse discourse roles, it regularizes role representations with SciEvent annotation-definition semantics using train-frequency-adaptive blending. The system is evaluated with the untouched official SciEvent evaluator on a frozen official-code split, with measured single-GPU train and inference memory under 30 GiB.

If exact paper split equivalence is not verified, do not replace “a frozen official-code split” with “the identical paper split”.

Do not claim that ModernBERT alone, Hungarian matching alone, LoRA, or context length is the novelty.

---

## 22. High-value failure diagnoses

If H0 is weak, inspect event-set matching, candidate recall, span mapping, and exporter correctness before changing the backbone.

If H0 is strong but H1 tuple anchoring is neutral, run the gold-tuple oracle. If the oracle is also neutral, tuple anchoring is probably unnecessary. If the oracle is strong but predicted-tuple H1 is weak, improve Action/Agent/Object prediction rather than adding unrelated modules.

If H2/H3 improves only ultra-rare roles with high variance and hurts overall Arg-C, do not promote it. If it improves rare roles with stable overall gains, inspect prototype cosine geometry and train-frequency dependence.

If rhetoric helps only one domain, test whether it is a robust discourse signal or a position/domain shortcut. Remove it if the gain is not stable.

If 1024 tokens does not help the audited long-window/long-distance subset, retain 512 for memory and speed.

If ModernBERT itself appears to be the bottleneck only after method components are stable, then run an optional matched-head DeBERTa-v3-large or SciBERT ablation. Do not do this before debugging the method.

If official-code regenerated split hashes vary across machines, first investigate Python hash seed/package/tokenizer differences. Do not compare unstable split runs.

---

## 23. Completion criteria

The workflow is complete when a fresh machine plus legally obtained CSCW abstracts can produce:

```text
pinned official repo commit
one frozen modern environment
pinned ModernBERT checkpoint revision
pinned preprocessing tokenizer revision
100% raw-abstract coverage audit
frozen official-code train/dev/test artifacts + SHA256
official evaluator SHA256 + passing synthetic metric tests
TARS source/configs
H0/H1/H2/H3 ablations
oracle diagnostics
multi-seed final checkpoints/predictions
official dev/test metrics
per-role/per-domain/error breakdowns
paired statistics for our internal ablations
measured train/inference VRAM <=30 GiB
paper-reported reference table marked reproduced=false
final comparison/SOTA-gate report with protocol caveats
```

There is intentionally **no baseline reproduction report** in the completion criteria.
