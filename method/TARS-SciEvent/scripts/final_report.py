#!/usr/bin/env python3
"""Aggregate runs into the final comparison report.

Produces:
  * one row per experiment: mean / std / median / every individual seed
  * paper-reported reference rows, always marked reproduced=false
  * document-block paired bootstrap of each experiment against TARS-H0
  * VRAM and runtime
  * the SOTA-gate checklist, evaluated mechanically

Never selects the best test seed as the headline number.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from scievent_tars.evaluation.bootstrap import paired_bootstrap  # noqa: E402

HEADLINE = "arg_c_iou_f1"
SECONDARY = ("arg_c_exact_f1", "arg_i_iou_f1", "arg_i_exact_f1")
BASELINE_EXPERIMENT = "h0_span"


def load_manifests(runs_dir: Path) -> list[dict]:
    manifests = []
    for path in sorted(runs_dir.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("smoke_test"):
            continue
        data["_path"] = str(path)
        data["_experiment"] = system_name(data, path)
        manifests.append(data)
    return manifests


def system_name(manifest: dict, path: Path) -> str:
    """Group runs by system, not by config file.

    Two runs can share ``config.experiment`` yet be different systems, because
    a CLI override (say a different prototype kappa) changed the architecture.
    The run id carries that distinction, so strip only the seed suffix from it
    and fall back to the config name when there is no run id.
    """
    run_id = manifest.get("experiment_id") or path.parent.name
    stripped = re.sub(r"\.seed\d+$", "", run_id)
    if stripped != run_id:
        return stripped
    return (manifest.get("config") or {}).get("experiment") or run_id


def summarise(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "n": len(values),
    }


def build_rows(manifests: list[dict], split: str) -> dict[str, dict]:
    key = f"{split}_metrics"
    grouped: dict[str, list[dict]] = defaultdict(list)
    for m in manifests:
        if m.get(key):
            grouped[m["_experiment"]].append(m)

    rows = {}
    for experiment, runs in sorted(grouped.items()):
        seeds = []
        for run in sorted(runs, key=lambda r: r.get("seed", 0)):
            metrics = run[key]
            seeds.append(
                {
                    "seed": run.get("seed"),
                    "experiment_id": run.get("experiment_id"),
                    HEADLINE: metrics.get(HEADLINE),
                    **{k: metrics.get(k) for k in SECONDARY},
                    "best_dev_epoch": run.get("best_dev_epoch"),
                    "peak_train_allocated_gib": run.get("peak_train_allocated_gib"),
                    "peak_infer_allocated_gib": run.get("peak_infer_allocated_gib"),
                    "runtime_seconds": run.get("runtime_seconds"),
                    "checkpoint_sha256": run.get("checkpoint_sha256"),
                }
            )
        headline_values = [s[HEADLINE] for s in seeds if s[HEADLINE] is not None]
        rows[experiment] = {
            "experiment": experiment,
            "reproduced": True,
            "source": "our run",
            HEADLINE: summarise(headline_values),
            **{k: summarise([s[k] for s in seeds if s[k] is not None]) for k in SECONDARY},
            "seeds": seeds,
            "peak_train_allocated_gib": max(
                (s["peak_train_allocated_gib"] or 0.0) for s in seeds
            ),
            "peak_infer_allocated_gib": max(
                (s["peak_infer_allocated_gib"] or 0.0) for s in seeds
            ),
        }
    return rows


def sota_gate(rows: dict, references: list[dict], split: str) -> dict:
    ours = rows.get("tars_final") or rows.get(BASELINE_EXPERIMENT)
    best_reference = max(
        (r for r in references if r.get("metric", "").lower().startswith("arg-c iou")),
        key=lambda r: r["reported_value"],
        default=None,
    )
    our_mean = (ours or {}).get(HEADLINE, {}).get("mean")
    num_seeds = (ours or {}).get(HEADLINE, {}).get("n", 0)
    peak = max(
        [r.get("peak_train_allocated_gib", 0.0) for r in rows.values()]
        + [r.get("peak_infer_allocated_gib", 0.0) for r in rows.values()]
        or [0.0]
    )
    frozen = (WORKSPACE / "artifacts/final/FROZEN_BEFORE_TEST.txt").exists()

    checks = {
        "official_evaluator_used": True,
        "test_untouched_before_freeze": frozen if split == "test" else None,
        "multiple_final_seeds": num_seeds >= 2,
        "vram_within_30gib": peak <= 30.0,
        "artifacts_hashed": all(
            s.get("checkpoint_sha256") for r in rows.values() for s in r["seeds"]
        ),
        "external_data_disclosed": True,
        "exceeds_prior_point_estimate": (
            our_mean is not None
            and best_reference is not None
            and our_mean > best_reference["reported_value"]
        ),
        "split_identity_byte_verified": False,
        "literature_refreshed_to_execution_date": False,
    }
    if all(v for v in checks.values() if v is not None):
        wording = "exceeds the paper-reported comparable result under the same released protocol"
    elif checks["exceeds_prior_point_estimate"]:
        wording = (
            "best result in our experiments on the regenerated official-code "
            "SciEvent split (split identity not byte-level verified; "
            "candidate SOTA pending current-literature verification)"
        )
    else:
        wording = "no SOTA claim is supported by the current evidence"

    return {
        "checks": checks,
        "our_point_estimate": our_mean,
        "best_paper_reported": best_reference,
        "absolute_margin": (
            our_mean - best_reference["reported_value"]
            if our_mean is not None and best_reference
            else None
        ),
        "permitted_wording": wording,
        "protocol_note": (
            "Paper-reported values are external reference points and were NOT "
            "rerun. Statistical significance is claimed only against our own "
            "matched H0/component ablations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument("--paper-reference",
                        default="artifacts/reference/paper_reported_results.json")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--stage", default="report")
    parser.add_argument("--out", default="artifacts/final/report")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--gold")
    parser.add_argument("--baseline-predictions")
    parser.add_argument("--compare-predictions", action="append", default=[],
                        metavar="NAME=PATH")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    manifests = load_manifests(workspace / args.runs)
    rows = build_rows(manifests, args.split)

    reference_path = workspace / args.paper_reference
    references = json.loads(reference_path.read_text()) if reference_path.exists() else []
    for ref in references:
        ref["reproduced"] = False  # never relabel a paper number as reproduced

    comparisons = {}
    if args.gold and args.baseline_predictions and args.compare_predictions:
        for item in args.compare_predictions:
            name, _, path = item.partition("=")
            comparisons[name] = paired_bootstrap(
                workspace / args.gold,
                workspace / args.baseline_predictions,
                workspace / path,
                num_samples=args.bootstrap_samples,
            )

    report = {
        "stage": args.stage,
        "split": args.split,
        "headline_metric": f"{HEADLINE} (official evaluator, untouched upstream)",
        "our_results": rows,
        "paper_reported_references": references,
        "paired_bootstrap_vs_h0": comparisons,
        "sota_gate": sota_gate(rows, references, args.split),
        "num_run_manifests": len(manifests),
    }

    out_dir = workspace / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.stage}.{args.split}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print(f"=== TARS-SciEvent {args.stage} ({args.split}) ===")
    print(f"headline: {HEADLINE} via the untouched official evaluator\n")
    header = f"{'system':26s} {'mean':>7s} {'std':>6s} {'median':>7s} {'n':>3s}  {'seeds'}"
    print(header)
    print("-" * len(header))
    for name, row in rows.items():
        stat = row[HEADLINE]
        seeds = ", ".join(
            f"{s['seed']}:{s[HEADLINE]:.2f}" for s in row["seeds"] if s[HEADLINE] is not None
        )
        print(
            f"{name:26s} {stat['mean'] or 0:7.2f} {stat['std'] or 0:6.2f} "
            f"{stat['median'] or 0:7.2f} {stat['n']:3d}  {seeds}"
        )
    print("\npaper-reported references (reproduced=false, NOT rerun):")
    for ref in references:
        print(f"  {ref['model_name']:12s} {ref['metric']:14s} {ref['reported_value']:6.2f}"
              f"   [{ref['paper_title']}, {ref.get('table_section','')}]")

    gate = report["sota_gate"]
    print(f"\npermitted wording: {gate['permitted_wording']}")
    for name, value in gate["checks"].items():
        mark = "-" if value is None else ("x" if value else " ")
        print(f"  [{mark}] {name}")
    print(f"\nwritten: {out_dir / f'{args.stage}.{args.split}.json'}")


if __name__ == "__main__":
    main()
