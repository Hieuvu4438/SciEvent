"""Error-analysis breakdowns for a promoted run.

These are **diagnostics**, not the headline metric. They reproduce the official
matching semantics (event-type sensitive, trigger-insensitive, greedy
one-to-one inside each window, Agent/PO/SO excluded) so that the per-bucket
numbers add up to the official totals, but the number quoted in any table is
always the one the untouched upstream evaluator printed.

One deliberate difference. Upstream accumulates matches into
``matched_set = set()`` keyed by the *(pred, gold) value tuples*, so when a
window contains two identical predictions matching two identical gold
arguments, the two pairs collapse into one set element and the match is
undercounted. We count matched *indices* instead, which is why a breakdown total
can sit a few tenths above the official number on windows with duplicate
(role, span) pairs. The official figure stays authoritative; this note exists so
the small gap is never mistaken for a bug in either implementation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..data.schema import SEMANTIC_ROLES, TRIGGER_COMPONENTS

EXCLUDED = set(TRIGGER_COMPONENTS)

#: Dimensions defined by the *gold* annotation, so a prediction cannot be placed
#: in them before it is matched. Only recall is meaningful for these; reporting a
#: precision or F1 of 0.0 would be an artifact, not a finding.
GOLD_ONLY_DIMENSIONS = {"same_role_multiplicity"}


def _load(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iou_match(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Upstream ``iou_overlap``: intersection / union > 0.5 (strict)."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return False
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union > 0.5


def exact_match(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a == b


def extract_semantic_arguments(record: dict) -> list[dict]:
    """One row per scored semantic argument of one window."""
    entity_map = {e["id"]: (e["start"], e["end"]) for e in record.get("entity_mentions", [])}
    rows: list[dict] = []
    for event in record.get("event_mentions", []):
        trigger = event["trigger"]
        for arg in event.get("arguments", []):
            role = arg["role"]
            if role in EXCLUDED:
                continue
            eid = arg.get("entity_id")
            if eid is not None and eid in entity_map:
                span = entity_map[eid]
            elif "start" in arg and "end" in arg:
                span = (arg["start"], arg["end"])
            else:
                continue
            rows.append(
                {
                    "event_type": event["event_type"],
                    "role": role,
                    "span": span,
                    "trigger": (trigger["start"], trigger["end"]),
                }
            )
    return rows


def _sentence_of(word: int, sentence_starts: list[int]) -> int:
    sent = 0
    for i, start in enumerate(sorted(sentence_starts)):
        if word >= start:
            sent = i
    return sent


def _width_bucket(width: int) -> str:
    for upper in (4, 8, 16, 32):
        if width <= upper:
            return f"<={upper}"
    return ">32"


def _length_bucket(num_words: int) -> str:
    for upper in (32, 64, 128):
        if num_words <= upper:
            return f"<={upper}"
    return ">128"


def _distance_bucket(distance: int) -> str:
    if distance == 0:
        return "same_sentence"
    if abs(distance) == 1:
        return "adjacent_sentence"
    return "distant_sentence"


def greedy_match(pred_rows: list[dict], gold_rows: list[dict], match_fn,
                 require_role: bool) -> tuple[set[int], set[int]]:
    """Upstream greedy one-to-one matching; returns matched pred/gold indices."""
    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    for pi, p in enumerate(pred_rows):
        for gi, g in enumerate(gold_rows):
            if gi in matched_gold:
                continue
            if p["event_type"] != g["event_type"]:
                continue
            if require_role and p["role"] != g["role"]:
                continue
            if match_fn(p["span"], g["span"]):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break
    return matched_pred, matched_gold


def compute_breakdowns(
    gold_path: str | Path, pred_path: str | Path, match_fn=iou_match
) -> dict:
    gold_records = _load(gold_path)
    pred_by_id = {
        (r.get("sent_id") or r["wnd_id"]): r for r in _load(pred_path)
    }

    buckets: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"matched": 0, "pred": 0, "gold": 0})
    )

    def add(dimension: str, key: str, field: str, amount: int = 1) -> None:
        buckets[dimension][str(key)][field] += amount

    for record in gold_records:
        sent_id = record.get("sent_id") or record["wnd_id"]
        domain = record["doc_id"].split("_")[0]
        num_words = len(record["tokens"])
        sentence_starts = record.get("sentence_starts") or [0]
        length_key = _length_bucket(num_words)

        gold_rows = extract_semantic_arguments(record)
        pred_rows = extract_semantic_arguments(pred_by_id.get(sent_id, {}))

        role_multiplicity = defaultdict(int)
        for row in gold_rows:
            role_multiplicity[row["role"]] += 1

        matched_pred, matched_gold = greedy_match(
            pred_rows, gold_rows, match_fn, require_role=True
        )

        for gi, row in enumerate(gold_rows):
            hit = int(gi in matched_gold)
            distance = _sentence_of(row["span"][0], sentence_starts) - _sentence_of(
                row["trigger"][0], sentence_starts
            )
            multiplicity = role_multiplicity[row["role"]]
            for dimension, key in (
                ("domain", domain),
                ("event_type", row["event_type"]),
                ("role", row["role"]),
                ("arg_width", _width_bucket(row["span"][1] - row["span"][0])),
                ("window_length", length_key),
                ("action_distance", _distance_bucket(distance)),
                ("cross_sentence", "cross" if distance != 0 else "single"),
                ("same_role_multiplicity", "1" if multiplicity == 1 else ">1"),
            ):
                add(dimension, key, "gold")
                add(dimension, key, "matched", hit)

        for pi, row in enumerate(pred_rows):
            distance = _sentence_of(row["span"][0], sentence_starts) - _sentence_of(
                row["trigger"][0], sentence_starts
            )
            for dimension, key in (
                ("domain", domain),
                ("event_type", row["event_type"]),
                ("role", row["role"]),
                ("arg_width", _width_bucket(row["span"][1] - row["span"][0])),
                ("window_length", length_key),
                ("action_distance", _distance_bucket(distance)),
                ("cross_sentence", "cross" if distance != 0 else "single"),
            ):
                add(dimension, key, "pred")

    out: dict[str, dict] = {}
    for dimension, keys in buckets.items():
        gold_only = dimension in GOLD_ONLY_DIMENSIONS
        out[dimension] = {}
        for key, counts in sorted(keys.items()):
            matched, pred, gold = counts["matched"], counts["pred"], counts["gold"]
            recall = matched / gold if gold else 0.0
            if gold_only:
                out[dimension][key] = {
                    "recall": 100 * recall,
                    "matched": matched,
                    "gold_total": gold,
                    "metric": "recall_only",
                    "note": "bucket is defined by gold; precision/F1 are undefined",
                }
                continue
            precision = matched / pred if pred else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            out[dimension][key] = {
                "precision": 100 * precision,
                "recall": 100 * recall,
                "f1": 100 * f1,
                "matched": matched,
                "pred_total": pred,
                "gold_total": gold,
            }
    return out


def boundary_error_report(gold_path: str | Path, pred_path: str | Path) -> dict:
    """Arg-I under EM vs IoU: a gap means loose boundaries, not better extraction."""
    report = {}
    for name, fn in (("exact", exact_match), ("iou", iou_match)):
        gold_records = _load(gold_path)
        pred_by_id = {(r.get("sent_id") or r["wnd_id"]): r for r in _load(pred_path)}
        matched = pred_total = gold_total = 0
        for record in gold_records:
            sent_id = record.get("sent_id") or record["wnd_id"]
            gold_rows = extract_semantic_arguments(record)
            pred_rows = extract_semantic_arguments(pred_by_id.get(sent_id, {}))
            mp, _ = greedy_match(pred_rows, gold_rows, fn, require_role=False)
            matched += len(mp)
            pred_total += len(pred_rows)
            gold_total += len(gold_rows)
        precision = matched / pred_total if pred_total else 0.0
        recall = matched / gold_total if gold_total else 0.0
        report[f"arg_i_{name}"] = {
            "precision": 100 * precision,
            "recall": 100 * recall,
            "f1": 100 * 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }
    report["boundary_only_gap_f1"] = (
        report["arg_i_iou"]["f1"] - report["arg_i_exact"]["f1"]
    )
    return report


def role_support_split(gold_path: str | Path, threshold: int = 100) -> dict[str, list[str]]:
    """Split roles into frequent / rare by gold support, for H3 reporting."""
    counts = defaultdict(int)
    for record in _load(gold_path):
        for row in extract_semantic_arguments(record):
            counts[row["role"]] += 1
    return {
        "frequent": sorted(r for r in SEMANTIC_ROLES if counts[r] >= threshold),
        "rare": sorted(r for r in SEMANTIC_ROLES if counts[r] < threshold),
        "support": dict(counts),
    }
