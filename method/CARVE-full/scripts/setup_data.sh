#!/usr/bin/env bash
# Fetch and prepare the SciEvent benchmark. Run once, from the repository root.
#
# CARVE never modifies the benchmark. This script only clones it and runs the
# benchmark's OWN preprocessing scripts to produce the ONEIE-format splits that
# every baseline in the paper is also trained and scored on.
set -euo pipefail

SCIEVENT_COMMIT="49cf9769c28f5ed8b04f61440b96163f50bca6f2"
DEST="third_party/SciEvent"

if [ ! -d "$DEST" ]; then
  echo "[1/4] Cloning SciEvent into $DEST"
  git clone https://github.com/desdai/SciEvent.git "$DEST"
  git -C "$DEST" checkout "$SCIEVENT_COMMIT"
else
  echo "[1/4] $DEST already present, skipping clone"
fi

cd "$DEST"

echo "[2/4] Building all_data.json"
bash data_scripts/shared/prepare_data.sh

echo "[3/4] Splitting into train/dev/test"
python3 data_scripts/shared/split_data.py

echo "[4/4] Renaming wnd_id -> sent_id for the ONEIE format"
bash data_scripts/ONEIE/wnd_id_rename.sh

cd - >/dev/null
echo
echo "Done. Verify with:  python3 -m pytest tests/ -q     (or: python3 tests/test_contract.py)"
echo
echo "NOTE: CSCW abstracts are not redistributable and are absent from the public"
echo "benchmark release. To reproduce the exact numbers in the paper you must"
echo "obtain those abstracts yourself; see the upstream README for details."
