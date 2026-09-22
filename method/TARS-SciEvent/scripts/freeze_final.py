#!/usr/bin/env python3
"""Freeze the winning configuration before any test evaluation (Stage E).

After this marker exists, no hyperparameter or architecture change may be
justified using test (STOP_TEST_01).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import yaml  # noqa: E402

from scievent_tars.training.registry import load_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the dev-selected winner")
    parser.add_argument("--out", default="artifacts/final/frozen_config.yaml")
    parser.add_argument("--dev-evidence", action="append", default=[],
                        help="run manifest(s) that justify this choice")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    out_path = workspace / args.out
    marker = out_path.parent / "FROZEN_BEFORE_TEST.txt"

    if marker.exists() and not args.force:
        raise SystemExit(
            f"{marker} already exists. Refusing to re-freeze; re-freezing after a "
            f"test read would invalidate the protocol. Use --force only if test "
            f"has provably not been evaluated."
        )

    resolved = load_config(workspace / args.config)
    resolved["_frozen"] = {
        "source_config": str(args.config),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dev_evidence": args.dev_evidence,
        "selection_metric": "dev/arg_c_iou_f1 (official evaluator)",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    digest = sha256_file(out_path)
    (out_path.parent / "frozen_config.sha256").write_text(
        f"{digest}  {out_path.name}\n", encoding="utf-8"
    )

    evidence = []
    for path in args.dev_evidence:
        manifest = workspace / path
        if manifest.exists():
            data = json.loads(manifest.read_text())
            evidence.append(
                {
                    "manifest": path,
                    "experiment_id": data.get("experiment_id"),
                    "seed": data.get("seed"),
                    "best_dev_epoch": data.get("best_dev_epoch"),
                    "dev_arg_c_iou_f1": data.get("dev_metrics", {}).get("arg_c_iou_f1"),
                }
            )

    try:
        overlay = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        overlay = None

    marker.write_text(
        json.dumps(
            {
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "frozen_config": str(out_path.relative_to(workspace)),
                "frozen_config_sha256": digest,
                "overlay_git_commit": overlay,
                "dev_evidence": evidence,
                "statement": (
                    "Test has not been evaluated before this timestamp. No "
                    "architecture or hyperparameter may be changed using test."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"frozen config : {out_path}")
    print(f"sha256        : {digest}")
    print(f"marker        : {marker}")


if __name__ == "__main__":
    main()
