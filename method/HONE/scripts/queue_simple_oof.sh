#!/bin/bash
# 5 out-of-fold CARVE-simple proposers (seed 42), two at a time.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
job() { python3 scripts/propose.py oof --fold $1 --seed 42 --proposer simple > logs/simple_oof_k$1_s42.log 2>&1
        echo "$(date +%H:%M) done k$1: $(grep -c '' data/cands/simple_oof_k$1_s42.jsonl 2>/dev/null || echo FAIL) windows"; }
job 0 & sleep 60; job 1 & wait
job 2 & sleep 60; job 3 & wait
job 4
echo "SIMPLE OOF COMPLETE"
