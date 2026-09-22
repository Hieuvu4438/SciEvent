#!/usr/bin/env python3
"""Phase 9 oracle diagnostics O1-O5 on train/dev only.

Oracle numbers bound what the rest of the pipeline could achieve; they never
enter a results table and never touch test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import torch  # noqa: E402

from predict_tars import build_loader, run_inference  # noqa: E402
from scievent_tars.data.reader import read_split  # noqa: E402
from scievent_tars.data.tokenizer_map import load_tokenizer  # noqa: E402
from scievent_tars.evaluation.export_official import export_predictions  # noqa: E402
from scievent_tars.evaluation.official_wrapper import score as official_score  # noqa: E402
from scievent_tars.evaluation.oracles import (  # noqa: E402
    ORACLE_DESCRIPTIONS,
    ORACLE_MODES,
    applicability,
)
from scievent_tars.training.registry import build_model, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev"],
                        help="oracles never run on test")
    parser.add_argument("--modes", default="O1,O2,O3,O4,O5")
    parser.add_argument("--tag", default="oracles")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    os.chdir(workspace)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples = read_split(config["data"]["train"])
    model, _ = build_model(config, train_examples, device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    )
    model.to(device)

    tokenizer = load_tokenizer(config["model"]["backbone_path"])
    examples = read_split(config["data"][args.split])
    loader = build_loader(
        examples,
        tokenizer,
        int(config["training"].get("max_length", 768)),
        int(config["training"].get("eval_batch_size", 8)),
        tokenizer.pad_token_id,
    )
    use_bf16 = bool(config["training"].get("bf16", True)) and torch.cuda.is_available()
    gold_path = workspace / config["data"][f"{args.split}_oneie"]
    evaluator = workspace / config["evaluation"]["official_evaluator"]
    out_dir = workspace / "artifacts" / "metrics" / args.tag

    results = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if mode not in ORACLE_MODES:
            raise SystemExit(f"unknown oracle mode {mode!r}")
        note = applicability(mode, model.config)
        predictions, diagnostics = run_inference(
            model, loader, device, use_bf16, oracle=ORACLE_MODES[mode]
        )
        pred_path = (
            workspace / "artifacts" / "predictions" / args.tag / f"{args.split}.{mode}.oneie.json"
        )
        export_predictions(gold_path, predictions, pred_path)
        payload = official_score(
            evaluator, gold_path, pred_path, require_exact_id_set=True,
            raw_stdout_path=out_dir / f"{args.split}.{mode}.official.txt",
        )
        results[mode] = {
            "description": ORACLE_DESCRIPTIONS[mode],
            "applicable": note.applicable,
            "applicability_note": note.reason,
            "metrics": payload["metrics"],
            "diagnostics": diagnostics,
            "is_oracle": mode != "O1",
        }
        flag = "" if note.applicable else "  [NOT INFORMATIVE FOR THIS CONFIG]"
        print(
            f"{mode}: arg_c_iou_f1={payload['metrics'].get('arg_c_iou_f1', 0.0):6.2f} "
            f"arg_i_iou_f1={payload['metrics'].get('arg_i_iou_f1', 0.0):6.2f} "
            f"| {ORACLE_DESCRIPTIONS[mode]}{flag}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "note": "ORACLE results never enter the SOTA/results table.",
        "results": results,
    }
    (out_dir / f"{args.split}.oracles.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwritten: artifacts/metrics/{args.tag}/{args.split}.oracles.json")


if __name__ == "__main__":
    main()
