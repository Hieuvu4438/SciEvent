"""Training loop, dev evaluation through the official evaluator, and the run
manifest contract.

Model selection is by **dev Arg-C IoU F1** as scored by the untouched upstream
evaluator. Internal metrics are debugging only.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.collator import LengthBucketSampler, WindowDataset, collate
from ..data.reader import read_split
from ..data.tokenizer_map import audit_lengths, load_tokenizer
from ..evaluation.export_official import export_predictions
from ..evaluation.official_wrapper import score as official_score
from ..modeling.model import PredictedEvent
from .losses import compute_total_loss
from .registry import build_event_cost_weights, build_loss_weights, build_model
from .scheduler import build_scheduler

HEADLINE_METRIC = "arg_c_iou_f1"
VRAM_HARD_GIB = 30.0
VRAM_TARGET_GIB = 28.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: str | Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


@dataclass
class VramReading:
    peak_allocated_gib: float
    peak_reserved_gib: float


def reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def read_vram() -> VramReading:
    if not torch.cuda.is_available():
        return VramReading(0.0, 0.0)
    return VramReading(
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )


class Trainer:
    def __init__(self, config: dict, seed: int, workspace: Path, experiment_id: str,
                 smoke_test: bool = False, smoke_windows: int = 24,
                 overfit: bool = False):
        self.config = config
        self.seed = seed
        self.workspace = Path(workspace)
        self.experiment_id = experiment_id
        self.smoke_test = smoke_test

        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = self.workspace / "artifacts" / "runs" / experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = config["data"]
        train_cfg = config["training"]
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg

        self.tokenizer = load_tokenizer(config["model"]["backbone_path"])
        self.max_length = int(train_cfg.get("max_length", 768))

        self.train_examples = read_split(self.workspace / data_cfg["train"])
        self.dev_examples = read_split(self.workspace / data_cfg["dev"])
        if smoke_test:
            self.train_examples = self.train_examples[:smoke_windows]
            self.dev_examples = self.dev_examples[:smoke_windows]
        if overfit:
            # Stage A acceptance: can the model fit 16-32 TRAIN windows at all?
            # "dev" here is the training subset itself and is never reported.
            self.dev_examples = list(self.train_examples)

        self.length_audit = audit_lengths(
            self.train_examples + self.dev_examples, self.tokenizer, self.max_length
        )

        self.model, self.capacity_info = build_model(
            config, self.train_examples, self.device
        )
        self.model.to(self.device)

        self.loss_weights = build_loss_weights(config)
        self.event_cost_weights = build_event_cost_weights(config)

        self.train_dataset = WindowDataset(self.train_examples, self.tokenizer, self.max_length)
        self.dev_dataset = WindowDataset(self.dev_examples, self.tokenizer, self.max_length)

        pad_id = self.tokenizer.pad_token_id
        self.micro_batch_size = int(train_cfg.get("micro_batch_size", 1))
        self.grad_accumulation = int(train_cfg.get("grad_accumulation", 16))

        self.train_sampler = LengthBucketSampler(
            self.train_dataset.piece_lengths(),
            batch_size=self.micro_batch_size,
            shuffle=True,
            seed=seed,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=lambda items: collate(items, pad_id),
            num_workers=0,
        )
        self.dev_loader = DataLoader(
            self.dev_dataset,
            batch_sampler=LengthBucketSampler(
                self.dev_dataset.piece_lengths(),
                batch_size=int(train_cfg.get("eval_batch_size", 4)),
                shuffle=False,
            ),
            collate_fn=lambda items: collate(items, pad_id),
            num_workers=0,
        )

        self.use_bf16 = bool(train_cfg.get("bf16", True)) and torch.cuda.is_available()
        self.autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float32

        self.optimizer = torch.optim.AdamW(
            self.model.parameter_groups(
                backbone_lr=float(train_cfg.get("backbone_lr", 1e-5)),
                head_lr=float(train_cfg.get("head_lr", 3e-4)),
                weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            ),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.max_epochs = int(train_cfg.get("max_epochs", 30)) if not smoke_test else int(
            train_cfg.get("smoke_epochs", 8)
        )
        steps_per_epoch = max(1, len(self.train_sampler) // self.grad_accumulation)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=steps_per_epoch * self.max_epochs,
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        )
        self.clip_grad_norm = float(train_cfg.get("clip_grad_norm", 1.0))
        self.early_stop_patience = int(train_cfg.get("early_stop_patience", 5))
        # Early stopping must not fire while the metric is still degenerate at
        # zero during LR warmup: a set-prediction model emits only NULL for the
        # first few epochs, so patience would expire before it ever predicts.
        self.early_stop_min_epochs = int(train_cfg.get("early_stop_min_epochs", 12))

        self.evaluator_path = self.workspace / config["evaluation"]["official_evaluator"]
        self.overfit = overfit
        self.dev_gold_oneie = self.workspace / data_cfg[
            "train_oneie" if overfit else "dev_oneie"
        ]

        self.history: list[dict] = []
        self.best_metric = -1.0
        self.best_epoch = -1
        self.peak_train = VramReading(0.0, 0.0)
        self.peak_infer = VramReading(0.0, 0.0)

    # -- training ---------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        totals: dict[str, float] = {}
        stat_totals: dict[str, float] = {}
        num_batches = 0
        start = time.time()

        for step, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.use_bf16):
                outputs = self.model(batch, training=True)
            anchor = self.model.prototypes.anchor_loss()
            breakdown = compute_total_loss(
                batch, outputs, self.model.layout, self.loss_weights,
                self.event_cost_weights, prototype_anchor=anchor,
            )
            (breakdown.total / self.grad_accumulation).backward()

            totals["total"] = totals.get("total", 0.0) + float(breakdown.total)
            for key, value in breakdown.parts.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            for key, value in breakdown.stats.items():
                stat_totals[key] = stat_totals.get(key, 0.0) + value
            num_batches += 1

            if (step + 1) % self.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        if num_batches % self.grad_accumulation != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "epoch": epoch,
            "seconds": time.time() - start,
            "loss": {k: v / max(num_batches, 1) for k, v in totals.items()},
            "supervision": stat_totals,
            "lr_backbone": self.optimizer.param_groups[0]["lr"],
            "lr_head": self.optimizer.param_groups[2]["lr"],
        }

    # -- inference --------------------------------------------------------------

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[dict[str, list[PredictedEvent]], dict]:
        self.model.eval()
        predictions: dict[str, list[PredictedEvent]] = {}
        recall_hits = 0
        recall_total = 0

        for batch in loader:
            batch = batch.to(self.device)
            with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.use_bf16):
                outputs = self.model(batch, training=False)
            decoded = self.model.decode(batch, outputs)
            for wnd_id, events in zip(batch.wnd_ids, decoded):
                predictions[wnd_id] = events
            for b, example in enumerate(batch.examples):
                proposed = set(outputs["candidates"].index_of[b])
                for span in example.all_semantic_spans():
                    recall_total += 1
                    recall_hits += int(span.as_tuple() in proposed)

        diagnostics = {
            "candidate_recall": recall_hits / recall_total if recall_total else 1.0,
            "num_gold_semantic_spans": recall_total,
        }
        return predictions, diagnostics

    def evaluate_dev(self, epoch: int) -> dict:
        reset_vram()
        predictions, diagnostics = self.predict(self.dev_loader)
        infer_vram = read_vram()
        self.peak_infer = VramReading(
            max(self.peak_infer.peak_allocated_gib, infer_vram.peak_allocated_gib),
            max(self.peak_infer.peak_reserved_gib, infer_vram.peak_reserved_gib),
        )

        pred_dir = self.workspace / "artifacts" / "predictions" / self.experiment_id
        pred_path = pred_dir / f"dev.epoch{epoch}.oneie.json"
        gold_path = self._gold_reference(pred_dir)
        export_info = export_predictions(gold_path, predictions, pred_path)

        metrics_dir = self.workspace / "artifacts" / "metrics" / self.experiment_id
        payload = official_score(
            self.evaluator_path,
            gold_path,
            pred_path,
            require_exact_id_set=True,
            raw_stdout_path=metrics_dir / f"dev.epoch{epoch}.official.txt",
            json_path=metrics_dir / f"dev.epoch{epoch}.official.json",
        )
        return {
            "epoch": epoch,
            "official": payload["metrics"],
            "export": export_info,
            "diagnostics": diagnostics,
            "pred_path": str(pred_path),
            "peak_infer_allocated_gib": infer_vram.peak_allocated_gib,
        }

    def _gold_reference(self, pred_dir: Path) -> Path:
        """Gold file restricted to the windows we actually evaluate."""
        wanted = {ex.wnd_id for ex in self.dev_examples}
        with open(self.dev_gold_oneie, "r", encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        if len(lines) == len(wanted):
            return self.dev_gold_oneie
        pred_dir.mkdir(parents=True, exist_ok=True)
        subset = pred_dir / "gold.subset.oneie.json"
        with open(subset, "w", encoding="utf-8") as out:
            for line in lines:
                record = json.loads(line)
                if (record.get("sent_id") or record["wnd_id"]) in wanted:
                    out.write(line)
        return subset

    # -- orchestration ----------------------------------------------------------

    def fit(self) -> dict:
        reset_vram()
        run_start = time.time()
        best_state_path = self.run_dir / "best.pt"
        patience = 0

        for epoch in range(1, self.max_epochs + 1):
            train_log = self.train_one_epoch(epoch)
            vram = read_vram()
            self.peak_train = VramReading(
                max(self.peak_train.peak_allocated_gib, vram.peak_allocated_gib),
                max(self.peak_train.peak_reserved_gib, vram.peak_reserved_gib),
            )

            dev_log = self.evaluate_dev(epoch)
            metric = dev_log["official"].get(HEADLINE_METRIC, 0.0)
            entry = {**train_log, "dev": dev_log, "dev_headline": metric,
                     "peak_train_allocated_gib": self.peak_train.peak_allocated_gib}
            self.history.append(entry)

            print(
                f"[epoch {epoch:02d}] loss={train_log['loss']['total']:.4f} "
                f"dev/{HEADLINE_METRIC}={metric:.2f} "
                f"dev/arg_c_exact_f1={dev_log['official'].get('arg_c_exact_f1', 0.0):.2f} "
                f"cand_recall={dev_log['diagnostics']['candidate_recall']:.4f} "
                f"vram={self.peak_train.peak_allocated_gib:.1f}GiB "
                f"({train_log['seconds']:.0f}s)",
                flush=True,
            )
            (self.run_dir / "history.json").write_text(
                json.dumps(self.history, indent=2), encoding="utf-8"
            )

            if metric > self.best_metric:
                self.best_metric = metric
                self.best_epoch = epoch
                patience = 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch,
                            "dev_headline": metric}, best_state_path)
            else:
                patience += 1
                if (
                    patience >= self.early_stop_patience
                    and epoch >= self.early_stop_min_epochs
                ):
                    print(f"early stop at epoch {epoch} (patience {patience})", flush=True)
                    break

        runtime = time.time() - run_start
        self._assert_vram()
        manifest = self._write_manifest(runtime, best_state_path)
        return manifest

    def _assert_vram(self) -> None:
        peak = max(self.peak_train.peak_allocated_gib, self.peak_infer.peak_allocated_gib)
        if peak > VRAM_HARD_GIB:
            raise RuntimeError(
                f"STOP_VRAM_01: peak allocated {peak:.2f} GiB exceeds {VRAM_HARD_GIB} GiB"
            )
        if peak > VRAM_TARGET_GIB:
            print(
                f"WARNING: peak allocated {peak:.2f} GiB exceeds the operational "
                f"target of {VRAM_TARGET_GIB} GiB",
                flush=True,
            )

    def _write_manifest(self, runtime: float, checkpoint_path: Path) -> dict:
        data_cfg = self.data_cfg
        split_hashes = {
            name: sha256_file(self.workspace / data_cfg[name])
            for name in ("train", "dev", "dev_oneie")
            if (self.workspace / data_cfg[name]).exists()
        }
        checkpoint_manifest_path = self.workspace / "checkpoints" / "manifest.json"
        backbone_revision = None
        if checkpoint_manifest_path.exists():
            entries = json.loads(checkpoint_manifest_path.read_text())
            entry = entries.get("modernbert-large", {})
            backbone_revision = entry.get("revision_sha") or entry.get("revision")

        upstream_file = self.workspace / "artifacts" / "audits" / "upstream_commit.txt"
        best_dev = next(
            (h["dev"]["official"] for h in self.history if h["epoch"] == self.best_epoch), {}
        )

        manifest = {
            "experiment_id": self.experiment_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "upstream_scievent_commit": (
                upstream_file.read_text().strip() if upstream_file.exists() else None
            ),
            "overlay_git_commit": git_commit(self.workspace),
            "data_protocol": "official_code_regenerated",
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "split_sha256": split_hashes,
            "evaluator_sha256": sha256_file(self.evaluator_path),
            "backbone_repo_id": self.config["model"].get("backbone_repo_id"),
            "backbone_revision_sha": backbone_revision,
            "training_regime": "full_ft",
            "external_supervised_data": False,
            "external_unlabeled_data": False,
            "seed": self.seed,
            "precision": "bf16" if self.use_bf16 else "fp32",
            "max_length": self.max_length,
            "micro_batch_size": self.micro_batch_size,
            "grad_accumulation": self.grad_accumulation,
            "optimizer": "AdamW",
            "backbone_lr": float(self.train_cfg.get("backbone_lr", 1e-5)),
            "head_lr": float(self.train_cfg.get("head_lr", 3e-4)),
            "epochs_run": len(self.history),
            "best_dev_epoch": self.best_epoch,
            "dev_metrics": best_dev,
            "test_metrics": {},
            "peak_train_allocated_gib": self.peak_train.peak_allocated_gib,
            "peak_train_reserved_gib": self.peak_train.peak_reserved_gib,
            "peak_infer_allocated_gib": self.peak_infer.peak_allocated_gib,
            "peak_infer_reserved_gib": self.peak_infer.peak_reserved_gib,
            "nvidia_smi_peak_mib": None,
            "runtime_seconds": runtime,
            "checkpoint_sha256": (
                sha256_file(checkpoint_path) if checkpoint_path.exists() else None
            ),
            "frozen_before_test_sha256": None,
            "length_audit": self.length_audit,
            "capacities": self.capacity_info["resolved_capacities"],
            "capacity_rules": self.capacity_info["capacity_evidence"]["rules"],
            "config": self.config,
            "smoke_test": self.smoke_test,
            "notes": "",
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return manifest
