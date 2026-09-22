#!/usr/bin/env python3
"""Train a TARS-SciEvent configuration on TRAIN and select on DEV.

Test is never touched here. Final test runs go through --final-test, which
requires the frozen-final marker to already exist (STOP_TEST_01).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from scievent_tars.training.registry import load_config  # noqa: E402
from scievent_tars.training.trainer import Trainer  # noqa: E402


def apply_override(config: dict, dotted_key: str, raw: str) -> None:
    """Apply a ``a.b.c=value`` CLI override, parsed as YAML scalar."""
    import yaml

    value = yaml.safe_load(raw)
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-windows", type=int, default=24)
    parser.add_argument("--overfit", action="store_true",
                        help="Stage A: fit a tiny TRAIN subset and score on itself")
    parser.add_argument("--experiment-id")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument("--final-test", action="store_true",
                        help="requires artifacts/final/FROZEN_BEFORE_TEST.txt")
    parser.add_argument("--override", action="append", default=[],
                        metavar="DOTTED.KEY=VALUE",
                        help="config override, e.g. training.backbone_lr=2e-5")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    os.chdir(workspace)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.final_test:
        marker = workspace / "artifacts" / "final" / "FROZEN_BEFORE_TEST.txt"
        if not marker.exists():
            raise SystemExit(
                "STOP_TEST_01: refusing to evaluate test before the frozen-final "
                f"marker exists at {marker}"
            )

    config = load_config(args.config)
    for item in args.override:
        key, _, raw = item.partition("=")
        apply_override(config, key.strip(), raw.strip())
    name = config.get("experiment", Path(args.config).stem)
    suffix = "overfit" if args.overfit else ("smoke" if args.smoke_test else f"seed{args.seed}")
    experiment_id = args.experiment_id or f"{name}.{suffix}"

    trainer = Trainer(
        config=config,
        seed=args.seed,
        workspace=workspace,
        experiment_id=experiment_id,
        smoke_test=args.smoke_test,
        smoke_windows=args.smoke_windows,
        overfit=args.overfit,
    )
    if args.override:
        print(f"overrides          : {args.override}")
    print(f"experiment_id      : {experiment_id}")
    print(f"resolved capacities: {json.dumps(trainer.capacity_info['resolved_capacities'])}")
    print(f"length audit       : {json.dumps(trainer.length_audit)}")

    manifest = trainer.fit()
    print(
        f"\nbest dev epoch {manifest['best_dev_epoch']} | "
        f"arg_c_iou_f1={manifest['dev_metrics'].get('arg_c_iou_f1', 0.0):.2f} | "
        f"arg_c_exact_f1={manifest['dev_metrics'].get('arg_c_exact_f1', 0.0):.2f} | "
        f"peak train {manifest['peak_train_allocated_gib']:.2f} GiB | "
        f"peak infer {manifest['peak_infer_allocated_gib']:.2f} GiB"
    )
    print(f"manifest: artifacts/runs/{experiment_id}/manifest.json")


if __name__ == "__main__":
    main()
