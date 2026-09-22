#!/usr/bin/env bash
set -euo pipefail

# scripts/setup_official_scievent.sh
# Execute official SciEvent data preprocessing, splitting, format conversion,
# and freeze into data/official/ with SHA256 checksums.

WORKSPACE="/home/haipd/SciEvent/method/TARS-SciEvent"
SCIEVENT_DIR="/home/haipd/SciEvent/third_party/SciEvent"
BART_DIR="${WORKSPACE}/checkpoints/preprocess-bart-tokenizer"
PYTHON="/home/haipd/miniconda3/envs/scievent-tars/bin/python"

echo "========================================================"
echo "Starting Official SciEvent Data Setup & Splitting"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

# Check hard gate: all 500 abstracts present
$PYTHON -c '
import json, glob, os
ann = "/home/haipd/SciEvent/third_party/SciEvent/SciEvent_data/annotated/event_extraction_finetune_model.jsonl"
seg = "/home/haipd/SciEvent/third_party/SciEvent/SciEvent_data/annotated/event_seg.jsonl"
abs_dir = "/home/haipd/SciEvent/third_party/SciEvent/SciEvent_data/abstracts_texts"
docs = set()
for p in (ann, seg):
    with open(p) as f:
        for l in f:
            docs.add(json.loads(l)["doc_id"])
existing = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(abs_dir, "*.txt")))
missing = docs - existing
print(f"Required: {len(docs)}, Existing: {len(existing)}, Missing: {len(missing)}")
if missing:
    raise RuntimeError(f"HARD GATE FAILED: {len(missing)} abstracts missing!")
print("HARD GATE PASS: 500/500 abstracts present.")
'

TMP_TEXTS=$(mktemp /tmp/scievent_texts_only.XXXXXX.jsonl)
trap 'rm -f "${TMP_TEXTS}"' EXIT

echo -e "\n[Step 1/5] Running prepare_segmentation.py..."
$PYTHON "${SCIEVENT_DIR}/data_scripts/shared/prepare_segmentation.py" \
  --annotation "${SCIEVENT_DIR}/SciEvent_data/annotated/event_extraction_finetune_model.jsonl" \
  --event_seg "${SCIEVENT_DIR}/SciEvent_data/annotated/event_seg.jsonl" \
  --abstract_dir "${SCIEVENT_DIR}/SciEvent_data/abstracts_texts" \
  --output "${TMP_TEXTS}" \
  --hf_model "${BART_DIR}" \
  --hf_cache "${WORKSPACE}/checkpoints/hf-cache"

echo -e "\n[Step 2/5] Running prepare_all_data.py..."
mkdir -p "${SCIEVENT_DIR}/SciEvent_data/DEGREE/processed"
$PYTHON "${SCIEVENT_DIR}/data_scripts/shared/prepare_all_data.py" \
  --annotation "${SCIEVENT_DIR}/SciEvent_data/annotated/event_extraction_finetune_model.jsonl" \
  --texts "${TMP_TEXTS}" \
  --output "${SCIEVENT_DIR}/SciEvent_data/DEGREE/processed/all_data.json"

echo -e "\n[Step 3/5] Running split_data.py with PYTHONHASHSEED=0..."
export PYTHONHASHSEED=0
cd "${SCIEVENT_DIR}"
$PYTHON "${SCIEVENT_DIR}/data_scripts/shared/split_data.py"

echo -e "\n[Step 4/5] Running wnd_id_rename.sh (ONEIE format conversion)..."
bash "${SCIEVENT_DIR}/data_scripts/ONEIE/wnd_id_rename.sh"

echo -e "\n[Step 5/5] Freezing official data into ${WORKSPACE}/data/official/..."
mkdir -p "${WORKSPACE}/data/official" "${WORKSPACE}/data/manifests"

cp "${SCIEVENT_DIR}/SciEvent_data/DEGREE/processed/all_data.json" "${WORKSPACE}/data/official/"
cp "${SCIEVENT_DIR}/SciEvent_data/DEGREE/all_splits/train.json" "${WORKSPACE}/data/official/"
cp "${SCIEVENT_DIR}/SciEvent_data/DEGREE/all_splits/dev.json" "${WORKSPACE}/data/official/"
cp "${SCIEVENT_DIR}/SciEvent_data/DEGREE/all_splits/test.json" "${WORKSPACE}/data/official/"
cp "${SCIEVENT_DIR}/SciEvent_data/ONEIE/all_splits/train.oneie.json" "${WORKSPACE}/data/official/"
cp "${SCIEVENT_DIR}/SciEvent_data/ONEIE/all_splits/dev.oneie.json" "${WORKSPACE}/data/official/"
cp "${SCIEVENT_DIR}/SciEvent_data/ONEIE/all_splits/test.oneie.json" "${WORKSPACE}/data/official/"

cd "${WORKSPACE}"
sha256sum data/official/* > data/manifests/official_files.sha256

echo -e "\n========================================================"
echo "Official data preparation completed successfully!"
echo "Generated files in data/official/:"
ls -lh data/official/
echo -e "\nChecksums (data/manifests/official_files.sha256):"
cat data/manifests/official_files.sha256
echo "========================================================"
