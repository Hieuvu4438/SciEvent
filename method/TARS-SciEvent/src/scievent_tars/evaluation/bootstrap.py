"""Document-block paired bootstrap and randomization tests.

Valid only between **our own** systems, because it needs both systems'
predictions on the same examples (e.g. TARS-FINAL vs TARS-H0, or
component-on vs component-off). Never run against paper-reported numbers.

Resampling is by ``doc_id`` block, since windows from one abstract are not
independent.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .breakdowns import extract_semantic_arguments, greedy_match, iou_match


def _load(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def per_window_counts(
    gold_path: str | Path, pred_path: str | Path, match_fn=iou_match,
    require_role: bool = True,
) -> dict[str, dict]:
    """``{wnd_id: {doc_id, matched, pred, gold}}`` for one system."""
    pred_by_id = {(r.get("sent_id") or r["wnd_id"]): r for r in _load(pred_path)}
    gold_records = _load(gold_path)
    gold_ids = {r.get("sent_id") or r["wnd_id"] for r in gold_records}
    if set(pred_by_id) != gold_ids:
        missing = sorted(gold_ids - set(pred_by_id))[:5]
        extra = sorted(set(pred_by_id) - gold_ids)[:5]
        raise ValueError(
            f"{pred_path}: prediction window set differs from gold "
            f"(missing={missing}, extra={extra}); paired statistics require the "
            f"systems to be scored on identical windows"
        )

    out: dict[str, dict] = {}
    for record in gold_records:
        sent_id = record.get("sent_id") or record["wnd_id"]
        gold_rows = extract_semantic_arguments(record)
        pred_rows = extract_semantic_arguments(pred_by_id.get(sent_id, {}))
        matched_pred, _ = greedy_match(pred_rows, gold_rows, match_fn, require_role)
        out[sent_id] = {
            "doc_id": record["doc_id"],
            "matched": len(matched_pred),
            "pred": len(pred_rows),
            "gold": len(gold_rows),
        }
    return out


def _f1(matched: float, pred: float, gold: float) -> float:
    precision = matched / pred if pred else 0.0
    recall = matched / gold if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def paired_bootstrap(
    gold_path: str | Path,
    pred_a: str | Path,
    pred_b: str | Path,
    num_samples: int = 10_000,
    seed: int = 12345,
    match_fn=iou_match,
    require_role: bool = True,
) -> dict:
    """Bootstrap ``F1(b) - F1(a)`` by resampling document blocks with replacement."""
    counts_a = per_window_counts(gold_path, pred_a, match_fn, require_role)
    counts_b = per_window_counts(gold_path, pred_b, match_fn, require_role)
    if set(counts_a) != set(counts_b):
        raise ValueError("the two systems were scored on different window sets")

    blocks: dict[str, list[str]] = defaultdict(list)
    for wnd_id, row in counts_a.items():
        blocks[row["doc_id"]].append(wnd_id)
    doc_ids = sorted(blocks)

    def totals(selected_docs) -> tuple[np.ndarray, np.ndarray]:
        a = np.zeros(3)
        b = np.zeros(3)
        for doc in selected_docs:
            for wnd_id in blocks[doc]:
                ra, rb = counts_a[wnd_id], counts_b[wnd_id]
                a += (ra["matched"], ra["pred"], ra["gold"])
                b += (rb["matched"], rb["pred"], rb["gold"])
        return a, b

    base_a, base_b = totals(doc_ids)
    observed = _f1(*base_b) - _f1(*base_a)

    rng = np.random.default_rng(seed)
    deltas = np.empty(num_samples)
    index = np.arange(len(doc_ids))
    for i in range(num_samples):
        picked = [doc_ids[j] for j in rng.choice(index, size=len(doc_ids), replace=True)]
        sample_a, sample_b = totals(picked)
        deltas[i] = _f1(*sample_b) - _f1(*sample_a)

    # Two-sided bootstrap p-value for H0: delta == 0.
    centred = deltas - deltas.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())

    return {
        "f1_a": 100 * _f1(*base_a),
        "f1_b": 100 * _f1(*base_b),
        "observed_delta_f1": 100 * observed,
        "ci95_low": 100 * float(np.percentile(deltas, 2.5)),
        "ci95_high": 100 * float(np.percentile(deltas, 97.5)),
        "bootstrap_p_value": p_value,
        "num_samples": num_samples,
        "num_document_blocks": len(doc_ids),
        "resampling_unit": "doc_id",
    }


def paired_randomization(
    gold_path: str | Path,
    pred_a: str | Path,
    pred_b: str | Path,
    num_samples: int = 10_000,
    seed: int = 12345,
    match_fn=iou_match,
    require_role: bool = True,
) -> dict:
    """Approximate randomization: swap whole document blocks between systems."""
    counts_a = per_window_counts(gold_path, pred_a, match_fn, require_role)
    counts_b = per_window_counts(gold_path, pred_b, match_fn, require_role)

    blocks: dict[str, list[str]] = defaultdict(list)
    for wnd_id, row in counts_a.items():
        blocks[row["doc_id"]].append(wnd_id)
    doc_ids = sorted(blocks)

    def block_totals(source, doc) -> np.ndarray:
        return np.sum(
            [[source[w]["matched"], source[w]["pred"], source[w]["gold"]]
             for w in blocks[doc]],
            axis=0,
        )

    block_a = np.stack([block_totals(counts_a, d) for d in doc_ids])
    block_b = np.stack([block_totals(counts_b, d) for d in doc_ids])

    observed = _f1(*block_b.sum(0)) - _f1(*block_a.sum(0))

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(num_samples):
        swap = rng.random(len(doc_ids)) < 0.5
        left = np.where(swap[:, None], block_b, block_a).sum(0)
        right = np.where(swap[:, None], block_a, block_b).sum(0)
        if abs(_f1(*right) - _f1(*left)) >= abs(observed):
            hits += 1

    return {
        "observed_delta_f1": 100 * observed,
        "randomization_p_value": (hits + 1) / (num_samples + 1),
        "num_samples": num_samples,
        "num_document_blocks": len(doc_ids),
    }
