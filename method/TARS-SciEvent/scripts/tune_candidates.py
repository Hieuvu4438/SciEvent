#!/usr/bin/env python3
"""Diagnose candidate-span proposal recall (STOP_MODEL_02).

The proposal is a non-parametric ranking on top of the trained boundary heads,
so the whole (k_start, k_end, k_span) grid can be scored from **one** forward
pass over the split. Reports the smallest configuration that clears the gate.

Run on train/dev only.
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
from scievent_tars.training.registry import build_model, load_config  # noqa: E402


@torch.no_grad()
def collect_boundary_logits(model, loader, device, use_bf16):
    """One forward pass; keep only what the proposal ranking needs."""
    model.eval()
    collected = []
    for batch in loader:
        batch = batch.to(device)
        word_states = model.encoder(
            batch.input_ids, batch.attention_mask, batch.piece_to_word,
            batch.word_mask.size(1),
        )
        if model.discourse is not None:
            word_states = model.discourse(
                word_states, batch.word_mask, batch.word_sentence_ids
            )
        start = model.candidates.start_head(word_states).squeeze(-1)
        end = model.candidates.end_head(word_states).squeeze(-1)
        start = start.masked_fill(~batch.word_mask, -1e4).float().cpu()
        end = end.masked_fill(~batch.word_mask, -1e4).float().cpu()
        for b, example in enumerate(batch.examples):
            n = int(batch.num_words[b])
            collected.append(
                {
                    "start": start[b, :n],
                    "end": end[b, :n],
                    "num_words": n,
                    "gold": [sp.as_tuple() for sp in example.all_semantic_spans()],
                }
            )
    return collected


def propose(start, end, num_words, k_start, k_end, k_span, max_width):
    ks = min(k_start, num_words)
    ke = min(k_end, num_words)
    top_starts = torch.topk(start, ks).indices
    top_ends = torch.topk(end, ke).indices
    gs = top_starts.unsqueeze(1).expand(ks, ke)
    ge = top_ends.unsqueeze(0).expand(ks, ke)
    width = ge - gs + 1
    valid = (width >= 1) & (width <= max_width)
    score = (start[gs] + end[ge]).masked_fill(~valid, float("-inf"))
    keep = min(k_span, int(valid.sum()))
    if keep <= 0:
        return set()
    flat = torch.topk(score.flatten(), keep).indices
    return set(zip(gs.flatten()[flat].tolist(), (ge.flatten()[flat] + 1).tolist()))


def evaluate(collected, k_start, k_end, k_span, max_width):
    hits = total = 0
    reachable = 0
    for item in collected:
        proposed = propose(
            item["start"], item["end"], item["num_words"],
            k_start, k_end, k_span, max_width,
        )
        for span in item["gold"]:
            total += 1
            hits += int(span in proposed)
            reachable += int(span[1] - span[0] <= max_width)
    return {
        "recall": hits / total if total else 1.0,
        "width_cap_ceiling": reachable / total if total else 1.0,
        "num_gold": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--gate", type=float, default=0.98)
    parser.add_argument("--k-boundary", default="64,96,128,192,256")
    parser.add_argument("--k-span", default="512,1024,2048,4096")
    parser.add_argument("--out", default="artifacts/audits/candidate_recall_sweep.json")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    os.chdir(workspace)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples = read_split(config["data"]["train"])
    model, capacity = build_model(config, train_examples, device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    )
    model.to(device)
    max_width = capacity["resolved_capacities"]["max_span_width"]

    tokenizer = load_tokenizer(config["model"]["backbone_path"])
    loader = build_loader(
        read_split(config["data"][args.split]), tokenizer,
        int(config["training"].get("max_length", 768)),
        int(config["training"].get("eval_batch_size", 8)), tokenizer.pad_token_id,
    )
    use_bf16 = bool(config["training"].get("bf16", True)) and torch.cuda.is_available()
    collected = collect_boundary_logits(model, loader, device, use_bf16)

    k_boundaries = [int(x) for x in args.k_boundary.split(",")]
    k_spans = [int(x) for x in args.k_span.split(",")]

    rows = []
    print(f"max_span_width={max_width}  split={args.split}  "
          f"gate={args.gate:.2f}  target={args.target:.2f}\n")
    header = f"{'k_start=k_end':>13s} {'k_span':>7s} {'recall':>8s} {'ceiling':>8s}  gate"
    print(header)
    print("-" * len(header))
    for kb in k_boundaries:
        for ks in k_spans:
            stats = evaluate(collected, kb, kb, ks, max_width)
            rows.append({"k_start": kb, "k_end": kb, "k_span": ks, **stats})
            mark = "PASS" if stats["recall"] >= args.gate else "FAIL"
            star = " <- target" if stats["recall"] >= args.target else ""
            print(f"{kb:13d} {ks:7d} {stats['recall']*100:7.2f}% "
                  f"{stats['width_cap_ceiling']*100:7.2f}%  {mark}{star}")

    passing = [r for r in rows if r["recall"] >= args.target]
    recommended = min(
        passing, key=lambda r: (r["k_span"], r["k_start"])
    ) if passing else None
    if recommended is None:
        passing = [r for r in rows if r["recall"] >= args.gate]
        recommended = min(passing, key=lambda r: (r["k_span"], r["k_start"])) if passing else None

    payload = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "max_span_width": max_width,
        "gate": args.gate,
        "target": args.target,
        "sweep": rows,
        "recommended": recommended,
    }
    out = workspace / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if recommended:
        print(f"\nrecommended: k_start=k_end={recommended['k_start']} "
              f"k_span={recommended['k_span']} -> {recommended['recall']*100:.2f}% recall")
    else:
        print("\nNo configuration clears the gate: the width cap itself is the "
              "bottleneck, so raise max_span_width instead.")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
