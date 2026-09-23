#!/usr/bin/env bash
# End-to-end reproduction of the CARVE results. Run from the repository root.
#
# Protocol: stages 1-3 use train + dev only. Stage 4 touches the test split
# exactly once, with checkpoints and the decoding rule already frozen.
set -euo pipefail

SEEDS=(42 13 101)

echo "=== Stage 1: correctness tests (data contract + official-scorer oracle) ==="
python3 tests/test_contract.py

echo
echo "=== Stage 2: train the frozen recipe on ${#SEEDS[@]} seeds (train -> dev) ==="
for SEED in "${SEEDS[@]}"; do
  python3 -m carve.train -c configs/carve.json --set run_name=carve_s${SEED} seed=${SEED}
done

echo
echo "=== Stage 3: freeze the decoding rule on DEV ONLY ==="
# tau / min_len are chosen to maximise the MEAN dev Arg-C IoU across seeds, so
# the rule is never fitted to whichever seed happens to look best.
python3 scripts/freeze_rules.py carve

echo
echo "=== Stage 4: FROZEN single evaluation on TEST (one fixed rule per seed) ==="
mkdir -p preds
for SEED in "${SEEDS[@]}"; do
  echo "--- seed ${SEED}"
  python3 -m carve.decode --ckpt runs/carve_s${SEED}/best.pt --split test \
    --rules assets/decoding_rules.json --out preds/test_s${SEED}.jsonl
done

echo
echo "=== Stage 5: result tables and error analysis ==="
python3 scripts/results_tables.py
python3 -m carve.analysis --pred preds/test_s13.jsonl --split test
