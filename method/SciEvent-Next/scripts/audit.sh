#!/usr/bin/env bash
# Reproduce the leakage / protocol audit in AUDIT.md. Run from method/SciEvent-Next/.
set -euo pipefail
UP=../../third_party/SciEvent
GOLD=$UP/SciEvent_data/ONEIE/all_splits

echo "=== A. official evaluator, run as a CLI, on our frozen test predictions ==="
python3 $UP/baselines/ONEIE/EM_overlap_eval.py \
  --pred artifacts/preds/test_preds_s13.jsonl --gold $GOLD/test.oneie.json \
  | grep -E "ROUGE-L OVERALL|Argument (Identification|Classification) \((Exact|IoU)\)"

echo
echo "=== A2. exporter oracle (gold spans -> our writer -> official evaluator) ==="
python3 tests/test_contract.py

echo
echo "=== B/C/G/H. split overlap, prediction sanity, structural assumptions ==="
python3 scripts/audit_checks.py

echo
echo "=== E. every load_split call in the training path ==="
grep -rn "load_split(" src/ scripts/ | grep -v "def load_split"

echo
echo "=== F. frozen rule must predate test predictions ==="
stat -c '%y  %n' artifacts/decoding_rules.json artifacts/preds/test_preds_s*.jsonl

echo
echo "=== third_party must be unmodified ==="
echo "modified files in third_party/SciEvent: $(cd $UP && git status --short | grep -c '^ M')"

echo
echo "=== D. document-disjoint robustness probe (rebuild + retrain, ~35 min) ==="
echo "    python3 scripts/make_docsplit.py"
echo "    for S in 42 13 101; do python3 src/scievent_next/train.py -c configs/docsplit.json \\"
echo "        --set run_name=docsplit_s\$S seed=\$S; done"
echo "    python3 scripts/freeze_rules.py docsplit \$PWD/artifacts/docsplit decoding_rules.docsplit.json"
echo "    for S in 42 13 101; do python3 src/scievent_next/infer.py --ckpt artifacts/runs/docsplit_s\$S/best.pt \\"
echo "        --split test --data_dir \$PWD/artifacts/docsplit --rules artifacts/decoding_rules.docsplit.json \\"
echo "        --out artifacts/preds/docsplit_test_s\$S.jsonl; done"
