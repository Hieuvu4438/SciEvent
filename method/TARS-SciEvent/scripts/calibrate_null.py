#!/usr/bin/env python3
"""Calibrate the single global NULL margin on dev.

The argument decoder chooses, per role slot, the best candidate span or NULL.
A positive ``null_bias`` makes the model more conservative, a negative one more
liberal. The plan allows exactly **one** global value tuned on dev -- never
per-role thresholds, because several roles have tiny dev support.

One forward pass covers the whole sweep: only the NULL column of the logits
changes, so decoding is replayed from cached outputs.

Dev only. Never run this against test.
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

from predict_tars import build_loader  # noqa: E402
from scievent_tars.data.reader import read_split  # noqa: E402
from scievent_tars.data.tokenizer_map import load_tokenizer  # noqa: E402
from scievent_tars.evaluation.export_official import export_predictions  # noqa: E402
from scievent_tars.evaluation.official_wrapper import score as official_score  # noqa: E402
from scievent_tars.training.registry import build_model, load_config  # noqa: E402

HEADLINE = "arg_c_iou_f1"


@torch.no_grad()
def sweep(model, loader, device, use_bf16, offsets):
    model.eval()
    predictions = {offset: {} for offset in offsets}
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(batch, training=False)
        base = outputs["arg_logits"]
        for offset in offsets:
            shifted = base.clone()
            shifted[..., -1] += offset
            decoded = model.decode(batch, {**outputs, "arg_logits": shifted})
            for wnd_id, events in zip(batch.wnd_ids, decoded):
                predictions[offset][wnd_id] = events
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offsets", default="-3,-2,-1.5,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--tag", default="null_calibration")
    parser.add_argument("--out", default="artifacts/audits/null_calibration.json")
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
    loader = build_loader(
        read_split(config["data"]["dev"]), tokenizer,
        int(config["training"].get("max_length", 768)),
        int(config["training"].get("eval_batch_size", 8)), tokenizer.pad_token_id,
    )
    use_bf16 = bool(config["training"].get("bf16", True)) and torch.cuda.is_available()

    offsets = [float(x) for x in args.offsets.split(",")]
    predictions = sweep(model, loader, device, use_bf16, offsets)

    gold = workspace / config["data"]["dev_oneie"]
    evaluator = workspace / config["evaluation"]["official_evaluator"]
    base_null_bias = float(config["model"].get("null_bias", 0.0))

    rows = []
    header = (f"{'offset':>7s} {'null_bias':>10s} {'argC_IoU':>9s} {'argC_EM':>8s} "
              f"{'argI_IoU':>9s} {'pred_args':>10s}")
    print(header)
    print("-" * len(header))
    for offset in offsets:
        pred_path = (workspace / "artifacts" / "predictions" / args.tag
                     / f"dev.null{offset:+g}.oneie.json")
        info = export_predictions(gold, predictions[offset], pred_path)
        metrics = official_score(evaluator, gold, pred_path,
                                 require_exact_id_set=True)["metrics"]
        row = {
            "offset": offset,
            "effective_null_bias": base_null_bias + offset,
            "num_predicted_semantic_arguments": info["num_predicted_semantic_arguments"],
            **{k: metrics.get(k) for k in
               (HEADLINE, "arg_c_exact_f1", "arg_i_iou_f1", "arg_c_iou_p", "arg_c_iou_r")},
        }
        rows.append(row)
        print(f"{offset:7.2f} {row['effective_null_bias']:10.2f} "
              f"{metrics.get(HEADLINE, 0):9.2f} {metrics.get('arg_c_exact_f1', 0):8.2f} "
              f"{metrics.get('arg_i_iou_f1', 0):9.2f} "
              f"{info['num_predicted_semantic_arguments']:10d}")

    best = max(rows, key=lambda r: (r[HEADLINE] or 0.0))
    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "base_null_bias": base_null_bias,
        "selection_metric": f"dev/{HEADLINE} (official evaluator)",
        "sweep": rows,
        "best": best,
        "note": "One global NULL margin, tuned on dev only. No per-role thresholds.",
    }
    out = workspace / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    gold_args = sum(
        len(ev.arguments) for ex in read_split(gold) for ev in ex.events
    )
    print(f"\ngold semantic arguments on dev: {gold_args}")
    print(f"best: null_bias={best['effective_null_bias']:+g} -> "
          f"{HEADLINE}={best[HEADLINE]:.2f} "
          f"(predicts {best['num_predicted_semantic_arguments']} args)")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
