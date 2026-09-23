#!/usr/bin/env bash
# Regenerate every table that appears in the paper, in order.
# Run from the repository root after training is complete.
set -euo pipefail

echo "################ Table 2 - main results vs published baselines ################"
python3 scripts/results_tables.py

echo
echo "################ Table 2b - bootstrap confidence intervals ####################"
python3 scripts/bootstrap_ci.py 2000 test

echo
echo "################ Table 3 - single-factor ablation #############################"
python3 scripts/ablation_table.py

echo
echo "################ Table 5 - controlled geometry test ###########################"
python3 scripts/geometry_control.py test

echo
echo "################ Table 6 - error taxonomy and length buckets ##################"
python3 -m carve.analysis --pred preds/test_s13.jsonl --split test

echo
echo "All paper tables regenerated."
