"""Strict wrapper around the untouched official evaluator.

We never reimplement headline scoring. This module validates the prediction
file, then subprocesses
``vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py`` and parses its stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from ..data.schema import EVENT_TYPES, SEMANTIC_ROLES, TRIGGER_COMPONENTS

VALID_ROLES = set(SEMANTIC_ROLES) | set(TRIGGER_COMPONENTS)

_SCORE_LINE = re.compile(
    r"^(?P<label>.+?)\s+-\s+P:\s*(?P<p>[\d.]+)\s*\(\d+/\d+\)\s+"
    r"R:\s*(?P<r>[\d.]+)\s*\(\d+/\d+\)\s+F1:\s*(?P<f>[\d.]+)\s*$"
)
_ROUGE_LINE = re.compile(
    r"^\[ROUGE-L OVERALL\] P:\s*(?P<p>[\d.]+)%,\s*R:\s*(?P<r>[\d.]+)%,\s*F1:\s*(?P<f>[\d.]+)%"
)


class PredictionContractError(RuntimeError):
    """STOP_METRIC_02 and friends: the prediction file is not scoreable."""


def _load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_predictions(pred_path: str | Path, gold_path: str | Path,
                         require_exact_id_set: bool = True) -> dict:
    pred = _load_jsonl(pred_path)
    gold = _load_jsonl(gold_path)

    def ids(records: list[dict]) -> list[str]:
        return [r.get("sent_id") or r["wnd_id"] for r in records]

    pred_ids, gold_ids = ids(pred), ids(gold)
    if len(set(pred_ids)) != len(pred_ids):
        raise PredictionContractError("duplicate sent_id in predictions")
    if require_exact_id_set and set(pred_ids) != set(gold_ids):
        missing = sorted(set(gold_ids) - set(pred_ids))[:5]
        extra = sorted(set(pred_ids) - set(gold_ids))[:5]
        raise PredictionContractError(
            f"STOP_METRIC_02: prediction id set != gold id set "
            f"(missing={missing}, extra={extra})"
        )

    gold_tokens = {gid: len(r["tokens"]) for gid, r in zip(gold_ids, gold)}

    seen_event_ids: set[str] = set()
    for record, sid in zip(pred, pred_ids):
        num_words = gold_tokens[sid]
        entity_map = {}
        for ent in record.get("entity_mentions", []):
            s, e = ent["start"], ent["end"]
            if not (isinstance(s, int) and isinstance(e, int)):
                raise PredictionContractError(f"{sid}: non-integer span ({s}, {e})")
            if not (0 <= s < e <= num_words):
                raise PredictionContractError(
                    f"{sid}: entity span ({s}, {e}) outside [0, {num_words}]"
                )
            entity_map[ent["id"]] = (s, e)

        for ev in record.get("event_mentions", []):
            if ev["event_type"] not in EVENT_TYPES:
                raise PredictionContractError(
                    f"{sid}: event type outside schema: {ev['event_type']!r}"
                )
            if ev["id"] in seen_event_ids:
                raise PredictionContractError(f"duplicate event id {ev['id']}")
            seen_event_ids.add(ev["id"])
            trig = ev["trigger"]
            if not (0 <= trig["start"] < trig["end"] <= num_words):
                raise PredictionContractError(
                    f"{sid}: trigger span ({trig['start']}, {trig['end']}) invalid"
                )
            for arg in ev.get("arguments", []):
                if arg["role"] not in VALID_ROLES:
                    raise PredictionContractError(
                        f"{sid}: role outside schema: {arg['role']!r}"
                    )
                eid = arg.get("entity_id")
                if eid is not None and eid not in entity_map:
                    if not ("start" in arg and "end" in arg):
                        raise PredictionContractError(
                            f"{sid}: argument references unknown entity {eid!r}"
                        )
                for key in ("start", "end"):
                    value = arg.get(key)
                    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                        raise PredictionContractError(f"{sid}: NaN/inf in argument span")

    return {
        "num_pred_windows": len(pred),
        "num_gold_windows": len(gold),
        "id_sets_equal": set(pred_ids) == set(gold_ids),
    }


def parse_evaluator_stdout(stdout: str) -> dict:
    """Extract the headline P/R/F1 rows from the official evaluator output."""
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        rouge = _ROUGE_LINE.match(line.strip())
        if rouge:
            metrics["rougeL_overall_p"] = float(rouge["p"])
            metrics["rougeL_overall_r"] = float(rouge["r"])
            metrics["rougeL_overall_f1"] = float(rouge["f"])
            continue
        match = _SCORE_LINE.match(line.rstrip())
        if not match:
            continue
        label = match["label"].strip()
        key = (
            label.lower()
            .replace("argument identification", "arg_i")
            .replace("argument classification", "arg_c")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )
        metrics[f"{key}_p"] = float(match["p"])
        metrics[f"{key}_r"] = float(match["r"])
        metrics[f"{key}_f1"] = float(match["f"])
    return metrics


def run_official_evaluator(
    evaluator_path: str | Path,
    gold_path: str | Path,
    pred_path: str | Path,
    python_executable: str | None = None,
) -> tuple[str, dict]:
    cmd = [
        python_executable or sys.executable,
        str(evaluator_path),
        "--pred",
        str(pred_path),
        "--gold",
        str(gold_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"official evaluator failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    return proc.stdout, parse_evaluator_stdout(proc.stdout)


def score(
    evaluator_path: str | Path,
    gold_path: str | Path,
    pred_path: str | Path,
    require_exact_id_set: bool = True,
    raw_stdout_path: str | Path | None = None,
    json_path: str | Path | None = None,
    python_executable: str | None = None,
) -> dict:
    contract = validate_predictions(pred_path, gold_path, require_exact_id_set)
    stdout, metrics = run_official_evaluator(
        evaluator_path, gold_path, pred_path, python_executable
    )

    payload = {
        "metrics": metrics,
        "contract": contract,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "pred_path": str(pred_path),
        "pred_sha256": sha256_file(pred_path),
    }

    if raw_stdout_path:
        Path(raw_stdout_path).parent.mkdir(parents=True, exist_ok=True)
        Path(raw_stdout_path).write_text(stdout, encoding="utf-8")
        payload["raw_stdout_path"] = str(raw_stdout_path)
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-evaluator", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--require-exact-id-set", action="store_true", default=True)
    parser.add_argument("--allow-id-mismatch", dest="require_exact_id_set",
                        action="store_false")
    parser.add_argument("--raw-stdout")
    parser.add_argument("--write-json")
    parser.add_argument("--python-executable")
    args = parser.parse_args()

    payload = score(
        args.official_evaluator,
        args.gold,
        args.pred,
        require_exact_id_set=args.require_exact_id_set,
        raw_stdout_path=args.raw_stdout,
        json_path=args.write_json,
        python_executable=args.python_executable,
    )
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
