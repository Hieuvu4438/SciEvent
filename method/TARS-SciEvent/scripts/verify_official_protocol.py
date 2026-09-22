#!/usr/bin/env python3
"""Lock the official protocol contract before any model work.

Checks, in order, the gates from the playbook:

  upstream repo pinned to the recorded commit and clean
  evaluator SHA256 unchanged
  frozen data file SHA256s match data/manifests/official_files.sha256
  data sanity (empty splits/spans/types/roles/ids)   -- plan 5.6
  raw abstract coverage 100%                         -- STOP_DATA_01
  gold -> official evaluator round-trip scores 100   -- STOP_METRIC_01

Exit code is non-zero if any gate fails; a report is always written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

PINNED_COMMIT = "49cf9769c28f5ed8b04f61440b96163f50bca6f2"

from scievent_tars.data.reader import read_split  # noqa: E402
from scievent_tars.data.schema import EVENT_TYPES, SEMANTIC_ROLES, TRIGGER_COMPONENTS  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail, stop_code: str | None = None) -> bool:
        self.checks.append(
            {"check": name, "ok": bool(ok), "detail": detail, "stop_code": stop_code}
        )
        status = "PASS" if ok else "FAIL"
        code = f" [{stop_code}]" if (stop_code and not ok) else ""
        print(f"[{status}]{code} {name}: {detail}")
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def check_upstream(workspace: Path, report: Report) -> None:
    vendor = workspace / "vendor" / "SciEvent"
    try:
        head = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(vendor), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except Exception as exc:
        report.add("upstream repo readable", False, str(exc))
        return
    report.add("upstream commit pinned", head == PINNED_COMMIT, head)

    # Untracked entries are expected and legitimate: the official preprocessing
    # writes its output into the vendor tree, and the licensed CSCW abstracts are
    # copied into SciEvent_data/abstracts_texts/ per plan 5.1. What must never
    # happen is a *tracked* upstream file being modified -- especially the
    # evaluator. Only tracked changes invalidate the protocol.
    tracked_changes = [l for l in porcelain if not l.startswith("??")]
    untracked = [l for l in porcelain if l.startswith("??")]
    report.add(
        "no tracked upstream file modified",
        not tracked_changes,
        tracked_changes[:10] or f"clean ({len(untracked)} untracked generated/licensed paths)",
    )


def check_evaluator(workspace: Path, report: Report) -> None:
    evaluator = workspace / "vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py"
    recorded = workspace / "artifacts/audits/official_evaluator.sha256"
    if not evaluator.exists():
        report.add("official evaluator present", False, str(evaluator))
        return
    digest = sha256_file(evaluator)
    if recorded.exists():
        expected = recorded.read_text().split()[0]
        report.add("evaluator sha256 unchanged", digest == expected,
                   f"{digest} (recorded {expected})")
    else:
        recorded.parent.mkdir(parents=True, exist_ok=True)
        recorded.write_text(f"{digest}  {evaluator}\n", encoding="utf-8")
        report.add("evaluator sha256 recorded", True, digest)


def check_data_hashes(workspace: Path, report: Report) -> None:
    manifest = workspace / "data/manifests/official_files.sha256"
    if not manifest.exists():
        report.add("data manifest present", False, str(manifest))
        return
    mismatched = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = workspace / name.strip()
        if not path.exists():
            mismatched.append(f"{name}: missing")
        elif sha256_file(path) != expected:
            mismatched.append(f"{name}: hash changed")
    report.add(
        "frozen data files match manifest", not mismatched,
        mismatched or f"{len(manifest.read_text().splitlines())} files verified",
        stop_code="STOP_DATA_03",
    )


def check_data_sanity(workspace: Path, report: Report) -> None:
    problems: list[str] = []
    summary: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        path = workspace / "data/official" / f"{split}.json"
        try:
            examples = read_split(path)  # already enforces span/role/type validity
        except Exception as exc:
            problems.append(f"{split}: {exc}")
            continue
        if not examples:
            problems.append(f"{split}: empty split")
            continue
        types = Counter(ev.event_type for ex in examples for ev in ex.events)
        roles = Counter(a.role for ex in examples for ev in ex.events for a in ev.arguments)
        for ex in examples:
            if not ex.words:
                problems.append(f"{split}/{ex.wnd_id}: empty reconstructed window")
            if not ex.wnd_id:
                problems.append(f"{split}: missing wnd_id")
        summary[split] = {
            "windows": len(examples),
            "events": sum(len(ex.events) for ex in examples),
            "event_types": dict(types),
            "semantic_roles": dict(roles),
            "max_events_per_window": max(len(ex.events) for ex in examples),
        }
        unknown_types = set(types) - set(EVENT_TYPES)
        unknown_roles = set(roles) - set(SEMANTIC_ROLES)
        if unknown_types:
            problems.append(f"{split}: event types outside schema {unknown_types}")
        if unknown_roles:
            problems.append(f"{split}: roles outside schema {unknown_roles}")

    report.add("data sanity (spans/types/roles/ids)", not problems,
               problems or summary, stop_code="STOP_DATA_02")

    oneie_ok = True
    details = {}
    for split in ("train", "dev", "test"):
        degree = workspace / "data/official" / f"{split}.json"
        oneie = workspace / "data/official" / f"{split}.oneie.json"
        if not oneie.exists():
            oneie_ok = False
            details[split] = "missing oneie file"
            continue
        a = {ex.wnd_id for ex in read_split(degree)}
        b = {ex.wnd_id for ex in read_split(oneie)}
        details[split] = "wnd_id set == sent_id set" if a == b else f"differ by {len(a ^ b)}"
        oneie_ok &= a == b
    report.add("wnd_id -> sent_id conversion consistent", oneie_ok, details)


def check_abstract_coverage(workspace: Path, report: Report) -> None:
    abstract_dir = workspace / "vendor/SciEvent/SciEvent_data/abstracts_texts"
    if not abstract_dir.exists():
        report.add("raw abstract directory present", False, str(abstract_dir),
                   stop_code="STOP_DATA_01")
        return
    available = {p.stem for p in abstract_dir.glob("*.txt")}
    needed = set()
    for split in ("train", "dev", "test"):
        needed |= {ex.doc_id for ex in read_split(workspace / "data/official" / f"{split}.json")}
    missing = sorted(needed - available)
    out = workspace / "artifacts/audits/missing_abstracts.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    report.add(
        "raw abstract coverage 100%", not missing,
        f"{len(needed) - len(missing)}/{len(needed)} docs covered"
        + (f"; missing written to {out.relative_to(workspace)}" if missing else ""),
        stop_code="STOP_DATA_01",
    )


def check_evaluator_roundtrip(workspace: Path, report: Report) -> None:
    from scievent_tars.evaluation.export_official import export_predictions
    from scievent_tars.evaluation.official_wrapper import score
    from scievent_tars.modeling.model import PredictedEvent

    evaluator = workspace / "vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py"
    for split in ("dev", "test"):
        gold = workspace / "data/official" / f"{split}.oneie.json"
        predictions = {
            ex.wnd_id: [
                PredictedEvent(
                    event_type=ev.event_type,
                    action_span=ev.action_span,
                    components={
                        "Agent": list(ev.agent_spans),
                        "PrimaryObject": list(ev.primary_object_spans),
                        "SecondaryObject": list(ev.secondary_object_spans),
                    },
                    arguments=[(a.role, a.span) for a in ev.arguments],
                )
                for ev in ex.events
            ]
            for ex in read_split(gold)
        }
        out = workspace / "artifacts/audits" / f"gold_roundtrip.{split}.oneie.json"
        export_predictions(gold, predictions, out)
        metrics = score(evaluator, gold, out, require_exact_id_set=True)["metrics"]
        perfect = all(
            abs(metrics.get(k, 0.0) - 100.0) < 1e-6
            for k in ("arg_c_exact_f1", "arg_c_iou_f1", "arg_i_exact_f1", "arg_i_iou_f1")
        )
        report.add(
            f"gold round-trip scores 100 on {split}", perfect,
            {k: metrics.get(k) for k in ("arg_c_exact_f1", "arg_c_iou_f1")},
            stop_code="STOP_METRIC_01",
        )
        out.unlink(missing_ok=True)


def check_role_definitions(workspace: Path, report: Report) -> None:
    import yaml

    path = workspace / "configs/tars/role_definitions.yaml"
    if not path.exists():
        report.add("role definitions present", False, str(path))
        return
    raw = yaml.safe_load(path.read_text())
    have_roles = set(raw.get("semantic_roles", {}))
    have_components = set(raw.get("trigger_components", {}))
    ok = have_roles == set(SEMANTIC_ROLES) and have_components == set(TRIGGER_COMPONENTS)
    report.add(
        "role definitions cover the official schema", ok,
        {"missing_roles": sorted(set(SEMANTIC_ROLES) - have_roles),
         "missing_components": sorted(set(TRIGGER_COMPONENTS) - have_components),
         "sha256": sha256_file(path)},
    )


def write_protocol_manifest(workspace: Path, path: Path, report: Report) -> None:
    """The reproducibility contract: what exactly this run's protocol consists of."""
    evaluator = workspace / "vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py"
    splits = {}
    for name in ("all_data", "train", "dev", "test",
                 "train.oneie", "dev.oneie", "test.oneie"):
        file = workspace / "data/official" / f"{name}.json"
        if file.exists():
            splits[file.name] = sha256_file(file)

    manifest = {
        "protocol": "official_code_regenerated",
        "protocol_note": (
            "Splits were regenerated with the official split_data.py under "
            "PYTHONHASHSEED=0. They are NOT asserted to be byte-identical to the "
            "unpublished paper split; these hashes are the reproducibility "
            "contract for our runs."
        ),
        "upstream_repo": "desdai/SciEvent",
        "upstream_branch": "EMNLP-2025",
        "upstream_commit_pinned": PINNED_COMMIT,
        "pythonhashseed": "0",
        "official_evaluator": {
            "path": "vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py",
            "sha256": sha256_file(evaluator) if evaluator.exists() else None,
            "modified_by_us": False,
        },
        "headline_metric": "arg_c_iou_f1 (Argument Classification, IoU > 0.5)",
        "iou_rule": "intersection / union > 0.5 (strict inequality)",
        "roles_excluded_from_semantic_score": sorted(TRIGGER_COMPONENTS),
        "event_types": list(EVENT_TYPES),
        "semantic_roles": list(SEMANTIC_ROLES),
        "split_sha256": splits,
        "role_definitions_sha256": (
            sha256_file(workspace / "configs/tars/role_definitions.yaml")
            if (workspace / "configs/tars/role_definitions.yaml").exists() else None
        ),
        "verification_passed": report.ok,
        "verified_checks": [c["check"] for c in report.checks if c["ok"]],
        "failed_checks": [c["check"] for c in report.checks if not c["ok"]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument("--out", default="artifacts/audits/protocol_verification.json")
    parser.add_argument("--manifest", default="data/manifests/protocol_manifest.json")
    parser.add_argument("--skip-abstracts", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    os.chdir(workspace)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    report = Report()
    check_upstream(workspace, report)
    check_evaluator(workspace, report)
    check_data_hashes(workspace, report)
    check_data_sanity(workspace, report)
    if not args.skip_abstracts:
        check_abstract_coverage(workspace, report)
    check_role_definitions(workspace, report)
    check_evaluator_roundtrip(workspace, report)

    out = workspace / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"ok": report.ok, "checks": report.checks}, indent=2, default=str),
        encoding="utf-8",
    )
    write_protocol_manifest(workspace, workspace / args.manifest, report)
    print(f"\nreport  : {args.out}")
    print(f"manifest: {args.manifest}")
    if not report.ok:
        failed = [c["check"] for c in report.checks if not c["ok"]]
        raise SystemExit(f"protocol verification FAILED: {failed}")
    print("protocol verification PASSED")


if __name__ == "__main__":
    main()
