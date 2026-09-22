#!/usr/bin/env bash
# End-to-end reproduction of SciEvent-Next. Run from method/SciEvent-Next/.
#
# Protocol: Stages 1-3 use train + dev only. Stage 4 touches test exactly once,
# with checkpoints and decoding rule already frozen by Stage 3.
set -euo pipefail

SRC=src/scievent_next

echo "=== Stage 1: correctness tests (data contract + official-evaluator oracle) ==="
python3 tests/test_contract.py

echo
echo "=== Stage 2: train the frozen recipe on 3 seeds (train -> dev) ==="
for SEED in 42 13 101; do
  python3 $SRC/train.py -c configs/final.json --set run_name=final_s${SEED} seed=${SEED}
done

echo
echo "=== Stage 3: freeze the decoding rule on DEV ONLY ==="
# Chooses tau/min_len by maximising the MEAN dev Arg-C IoU across seeds, so the
# rule is not fitted to whichever seed happens to look best.
python3 scripts/freeze_rules.py

echo
echo "=== Stage 4: FROZEN single evaluation on TEST (3 seeds, one fixed rule) ==="
for SEED in 42 13 101; do
  echo "--- seed ${SEED}"
  python3 $SRC/infer.py --ckpt artifacts/runs/final_s${SEED}/best.pt --split test \
    --rules artifacts/decoding_rules.json --out artifacts/preds/test_preds_s${SEED}.jsonl
done

echo
echo "=== Stage 5: error analysis + experiment registry ==="
python3 $SRC/diagnose.py --pred artifacts/preds/test_preds_s13.jsonl --split test
python3 scripts/registry.py
