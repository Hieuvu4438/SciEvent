#!/usr/bin/env bash
# Reproduce the leakage and protocol audit (docs/PAPER_VI.md, section 8).
set -euo pipefail

SCIEVENT="${SCIEVENT_ROOT:-third_party/SciEvent}"
GOLD="$SCIEVENT/SciEvent_data/ONEIE/all_splits"

echo "=== 8.1 official scorer, run as a CLI, on our frozen test predictions ==="
python3 "$SCIEVENT/baselines/ONEIE/EM_overlap_eval.py" \
  --pred preds/test_s13.jsonl --gold "$GOLD/test.oneie.json" \
  | grep -E "ROUGE-L OVERALL|Argument (Identification|Classification) \((Exact|IoU)\)"

echo
echo "=== exporter oracle (gold spans -> our writer -> official scorer) ==="
python3 tests/test_contract.py

echo
echo "=== 8.2/8.3 split overlap, prediction sanity, structural assumptions ==="
python3 scripts/audit_checks.py

echo
echo "=== every data-loading call in the training path ==="
grep -rn "load_split(" carve/ scripts/ | grep -v "def load_split"

echo
echo "=== the frozen rule must predate the test predictions ==="
stat -c '%y  %n' assets/decoding_rules.json preds/test_s*.jsonl 2>/dev/null || true

echo
echo "=== the benchmark checkout must be unmodified ==="
echo "modified files in $SCIEVENT: $(git -C "$SCIEVENT" status --short 2>/dev/null | grep -c '^ M' || echo 0)"
