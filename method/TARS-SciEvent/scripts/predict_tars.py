#!/usr/bin/env python3
"""Run inference from a trained checkpoint and score with the official evaluator.

``--split test`` is refused unless the frozen-final marker exists (STOP_TEST_01).
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

from scievent_tars.data.collator import LengthBucketSampler, WindowDataset, collate  # noqa: E402
from scievent_tars.data.reader import read_split  # noqa: E402
from scievent_tars.data.tokenizer_map import load_tokenizer  # noqa: E402
from scievent_tars.evaluation.breakdowns import boundary_error_report, compute_breakdowns  # noqa: E402
from scievent_tars.evaluation.export_official import export_predictions  # noqa: E402
from scievent_tars.evaluation.official_wrapper import score as official_score  # noqa: E402
from scievent_tars.training.registry import build_model, load_config  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def build_loader(examples, tokenizer, max_length, batch_size, pad_id):
    dataset = WindowDataset(examples, tokenizer, max_length)
    sampler = LengthBucketSampler(dataset.piece_lengths(), batch_size, shuffle=False)
    return DataLoader(
        dataset, batch_sampler=sampler, collate_fn=lambda items: collate(items, pad_id)
    )


@torch.no_grad()
def run_inference(model, loader, device, use_bf16, oracle=None):
    model.eval()
    predictions = {}
    recall_hits = recall_total = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(batch, training=False, oracle=oracle)
        for wnd_id, events in zip(batch.wnd_ids, model.decode(batch, outputs, oracle=oracle)):
            predictions[wnd_id] = events
        for b, example in enumerate(batch.examples):
            proposed = set(outputs["candidates"].index_of[b])
            for span in example.all_semantic_spans():
                recall_total += 1
                recall_hits += int(span.as_tuple() in proposed)
    return predictions, {
        "candidate_recall": recall_hits / recall_total if recall_total else 1.0,
        "num_gold_semantic_spans": recall_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--out-dir")
    parser.add_argument("--tag", default="predict")
    parser.add_argument("--breakdowns", action="store_true")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    os.chdir(workspace)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.split == "test":
        marker = workspace / "artifacts" / "final" / "FROZEN_BEFORE_TEST.txt"
        if not marker.exists():
            raise SystemExit(f"STOP_TEST_01: no frozen-final marker at {marker}")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = read_split(config["data"]["train"])
    model, capacity_info = build_model(config, train_examples, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    tokenizer = load_tokenizer(config["model"]["backbone_path"])
    max_length = int(config["training"].get("max_length", 768))
    examples = read_split(config["data"][args.split])
    loader = build_loader(
        examples, tokenizer, max_length,
        int(config["training"].get("eval_batch_size", 8)), tokenizer.pad_token_id,
    )

    use_bf16 = bool(config["training"].get("bf16", True)) and torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    predictions, diagnostics = run_inference(model, loader, device, use_bf16)
    peak_infer = (
        torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    )

    out_dir = Path(args.out_dir or workspace / "artifacts" / "predictions" / args.tag)
    pred_path = out_dir / f"{args.split}.oneie.json"
    gold_path = workspace / config["data"][f"{args.split}_oneie"]
    export_info = export_predictions(gold_path, predictions, pred_path)

    metrics_dir = workspace / "artifacts" / "metrics" / args.tag
    payload = official_score(
        workspace / config["evaluation"]["official_evaluator"],
        gold_path,
        pred_path,
        require_exact_id_set=True,
        raw_stdout_path=metrics_dir / f"{args.split}.official.txt",
        json_path=metrics_dir / f"{args.split}.official.json",
    )
    payload["diagnostics"] = diagnostics
    payload["export"] = export_info
    payload["peak_infer_allocated_gib"] = peak_infer
    payload["capacities"] = capacity_info["resolved_capacities"]

    if args.breakdowns:
        payload["breakdowns"] = compute_breakdowns(gold_path, pred_path)
        payload["boundary_errors"] = boundary_error_report(gold_path, pred_path)
        (metrics_dir / f"{args.split}.breakdowns.json").write_text(
            json.dumps(
                {"breakdowns": payload["breakdowns"],
                 "boundary_errors": payload["boundary_errors"]},
                indent=2,
            ),
            encoding="utf-8",
        )

    (metrics_dir / f"{args.split}.summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metrics": payload["metrics"], "diagnostics": diagnostics,
                      "peak_infer_allocated_gib": peak_infer}, indent=2))
    if peak_infer > 30.0:
        raise SystemExit(f"STOP_VRAM_01: inference peak {peak_infer:.2f} GiB > 30 GiB")


if __name__ == "__main__":
    main()
