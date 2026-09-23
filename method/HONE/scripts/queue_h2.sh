#!/bin/bash
# Wait for all out-of-fold proposals, then train H2 verifiers (pooled 3 proposer seeds).
cd "$(dirname "$0")/.."
OUT=$1
until grep -q "OOF QUEUE COMPLETE" "$OUT"; do sleep 30; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
run() { python3 scripts/train_verifier.py --name h2_v$1_p3 --prop_seeds 42 13 101 --seed $1 \
          --batch_size 8 --accum 2 --eval_batch_size 16 --epochs 5 --grad_ckpt > logs/h2_v$1_p3.log 2>&1; }
run 42 & sleep 120; run 13 & wait
run 101
echo "H2 QUEUE COMPLETE"
