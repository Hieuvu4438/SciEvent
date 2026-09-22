#!/usr/bin/env python3
"""Measure peak train and inference VRAM for a configuration.

Reports both the internal torch counters and an external nvidia-smi poll.
Hard gate: peak allocated <= 30 GiB (STOP_VRAM_01); operational target 28 GiB.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scievent_tars.data.collator import LengthBucketSampler, WindowDataset, collate  # noqa: E402
from scievent_tars.data.reader import read_split  # noqa: E402
from scievent_tars.data.tokenizer_map import load_tokenizer  # noqa: E402
from scievent_tars.training.losses import compute_total_loss  # noqa: E402
from scievent_tars.training.registry import (  # noqa: E402
    build_event_cost_weights,
    build_loss_weights,
    build_model,
    load_config,
)

HARD_LIMIT_GIB = 30.0
TARGET_GIB = 28.0


class SmiPoller(threading.Thread):
    """External nvidia-smi poll, since torch counters miss allocator overhead."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()

    def run(self) -> None:
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2 and parts[0] == pid:
                        self.peak_mib = max(self.peak_mib, int(parts[1]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> int:
        self._stop.set()
        self.join(timeout=5)
        return self.peak_mib


def reset() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peaks() -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated() / 2**30,
        torch.cuda.max_memory_reserved() / 2**30,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", default="artifacts/audits/vram_measurement.json")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    os.chdir(workspace)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; VRAM measurement is meaningless")

    config = load_config(args.config)
    device = torch.device("cuda")
    train_examples = read_split(config["data"]["train"])
    model, _ = build_model(config, train_examples, device)
    model.to(device)

    tokenizer = load_tokenizer(config["model"]["backbone_path"])
    max_length = int(config["training"].get("max_length", 768))
    train_cfg = config["training"]

    # Worst case for memory is the longest windows, so sort descending.
    longest = sorted(train_examples, key=lambda e: e.num_words, reverse=True)
    subset = longest[: max(args.steps * int(train_cfg.get("micro_batch_size", 4)), 8)]
    dataset = WindowDataset(subset, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_sampler=LengthBucketSampler(
            dataset.piece_lengths(), int(train_cfg.get("micro_batch_size", 4)), shuffle=False
        ),
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )

    weights = build_loss_weights(config)
    cost_weights = build_event_cost_weights(config)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            float(train_cfg.get("backbone_lr", 1e-5)),
            float(train_cfg.get("head_lr", 3e-4)),
            float(train_cfg.get("weight_decay", 0.01)),
        )
    )
    use_bf16 = bool(train_cfg.get("bf16", True))

    poller = SmiPoller()
    poller.start()

    # --- training peak ---
    model.train()
    reset()
    start = time.time()
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(batch, training=True)
        breakdown = compute_total_loss(
            batch, outputs, model.layout, weights, cost_weights,
            prototype_anchor=model.prototypes.anchor_loss(),
        )
        breakdown.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    train_alloc, train_reserved = peaks()
    train_seconds = time.time() - start

    # --- inference peak ---
    model.eval()
    reset()
    start = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= args.steps:
                break
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(batch, training=False)
            model.decode(batch, outputs)
    infer_alloc, infer_reserved = peaks()
    infer_seconds = time.time() - start

    smi_peak_mib = poller.stop()
    worst = max(train_alloc, infer_alloc)

    result = {
        "config": args.config,
        "micro_batch_size": int(train_cfg.get("micro_batch_size", 4)),
        "max_length": max_length,
        "longest_window_words": subset[0].num_words,
        "peak_train_allocated_gib": train_alloc,
        "peak_train_reserved_gib": train_reserved,
        "peak_infer_allocated_gib": infer_alloc,
        "peak_infer_reserved_gib": infer_reserved,
        "nvidia_smi_peak_mib": smi_peak_mib,
        "nvidia_smi_peak_gib": smi_peak_mib / 1024,
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "hard_limit_gib": HARD_LIMIT_GIB,
        "target_gib": TARGET_GIB,
        "within_hard_limit": worst <= HARD_LIMIT_GIB,
        "within_target": worst <= TARGET_GIB,
    }
    out = workspace / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    if worst > HARD_LIMIT_GIB:
        raise SystemExit(f"STOP_VRAM_01: peak allocated {worst:.2f} GiB > {HARD_LIMIT_GIB}")
    if worst > TARGET_GIB:
        print(f"WARNING: peak {worst:.2f} GiB exceeds the {TARGET_GIB} GiB target")


if __name__ == "__main__":
    main()
