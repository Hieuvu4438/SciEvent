#!/usr/bin/env bash
# Machine audit + protocol gate. Run before any training.
#
# Records hardware/environment provenance, then runs the full protocol
# verification and the test suite. Non-zero exit means do not train.
set -euo pipefail

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$WORKSPACE"

PYTHON="${PYTHON:-$(command -v python)}"
mkdir -p artifacts/audits

echo "=== [1/4] hardware + environment audit ==="
{
  date -u
  uname -a
  nvidia-smi || true
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
  "$PYTHON" --version || true
  conda --version || true
  git --version
} 2>&1 | tee artifacts/audits/hardware.txt

echo
echo "=== [2/4] pinned upstream commit ==="
git -C vendor/SciEvent rev-parse HEAD | tee artifacts/audits/upstream_commit.txt

echo
echo "=== [3/4] official protocol verification ==="
"$PYTHON" scripts/verify_official_protocol.py --workspace "$WORKSPACE"

echo
echo "=== [4/4] test suite (metric contract, span round-trip, export) ==="
"$PYTHON" -m pytest -q \
  tests/test_metric_contract.py \
  tests/test_span_roundtrip.py \
  tests/test_multi_event_window.py \
  tests/test_candidate_recall.py \
  tests/test_official_export.py

"$PYTHON" -m pip freeze > artifacts/audits/pip_freeze.txt

echo
echo "PREFLIGHT PASSED -- safe to train."
