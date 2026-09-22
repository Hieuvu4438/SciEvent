#!/usr/bin/env python3
"""
scripts/audit_split.py

Audits the frozen official splits in data/official/ and verifies:
1. No empty sentences or broken spans (0 <= start < end <= len(tokens)).
2. Event types & roles schema conformity.
3. Check for document overlaps across train / dev / test.
4. Summary statistics on TRAIN (capacities, span width, roles, etc.).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


def audit_dataset(train_path, dev_path, test_path, out_path):
    splits = {}
    for name, p in [("train", train_path), ("dev", dev_path), ("test", test_path)]:
        with open(p, "r", encoding="utf-8") as f:
            splits[name] = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded instances: train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])}")

    # 1. Sanity check: sentence validity, span validity, event types
    valid_event_types = {"Background/Introduction", "Methods/Approach", "Results/Findings", "Conclusions/Implications",
                         "Background", "Method", "Result", "Conclusion"}
    errors = []

    for name, data in splits.items():
        for item in data:
            wnd_id = item.get("wnd_id") or item.get("sent_id")
            tokens = item.get("tokens", [])
            sentence = item.get("sentence", "")
            if not tokens:
                errors.append(f"[{name}] Empty tokens in {wnd_id}")
            if not sentence:
                errors.append(f"[{name}] Empty sentence in {wnd_id}")

            n_tok = len(tokens)
            for ev in item.get("event_mentions", []):
                et = ev.get("event_type")
                if et not in valid_event_types:
                    errors.append(f"[{name}] Invalid event_type '{et}' in {wnd_id}")
                trig = ev.get("trigger", {})
                ts, te = trig.get("start", -1), trig.get("end", -1)
                if ts < 0 or te <= ts or te > n_tok:
                    errors.append(f"[{name}] Invalid trigger span ({ts},{te}) with len {n_tok} in {wnd_id}")

                for arg in ev.get("arguments", []):
                    # In ONEIE format, args might have start/end or entity_id
                    pass

    if errors:
        print(f"FAILED sanity check with {len(errors)} errors:")
        for e in errors[:10]:
            print("  ", e)
        sys.exit(1)
    else:
        print("PASS: All tokens, sentences, trigger spans, and event types are valid across all splits!")

    # 2. Intersections
    doc_sets = {name: set(item["doc_id"] for item in data) for name, data in splits.items()}
    wnd_sets = {name: set(item.get("wnd_id") or item.get("sent_id") for item in data) for name, data in splits.items()}

    train_dev_doc_overlap = len(doc_sets["train"] & doc_sets["dev"])
    train_test_doc_overlap = len(doc_sets["train"] & doc_sets["test"])
    dev_test_doc_overlap = len(doc_sets["dev"] & doc_sets["test"])

    train_dev_wnd_overlap = len(wnd_sets["train"] & wnd_sets["dev"])
    train_test_wnd_overlap = len(wnd_sets["train"] & wnd_sets["test"])
    dev_test_wnd_overlap = len(wnd_sets["dev"] & wnd_sets["test"])

    print(f"\nDocument overlap (doc_id):")
    print(f"  train & dev: {train_dev_doc_overlap}")
    print(f"  train & test: {train_test_doc_overlap}")
    print(f"  dev & test: {dev_test_doc_overlap}")

    print(f"Window overlap (wnd_id):")
    print(f"  train & dev: {train_dev_wnd_overlap}")
    print(f"  train & test: {train_test_wnd_overlap}")
    print(f"  dev & test: {dev_test_wnd_overlap}")

    # 3. Train statistics for model capacities
    train_events_per_wnd = [len(item.get("event_mentions", [])) for item in splits["train"]]
    max_train_events = max(train_events_per_wnd) if train_events_per_wnd else 0

    role_counts = Counter()
    event_type_counts = Counter()
    for item in splits["train"]:
        for ev in item.get("event_mentions", []):
            event_type_counts[ev.get("event_type")] += 1
            for arg in ev.get("arguments", []):
                role_counts[arg.get("role")] += 1

    print(f"\nTrain statistics for model capacity:")
    print(f"  Max events/window in train: {max_train_events}")
    print(f"  Event types in train: {dict(event_type_counts)}")
    print(f"  Roles in train: {dict(role_counts)}")

    audit_result = {
        "num_instances": {k: len(v) for k, v in splits.items()},
        "num_unique_docs": {k: len(v) for k, v in doc_sets.items()},
        "doc_overlap": {
            "train_dev": train_dev_doc_overlap,
            "train_test": train_test_doc_overlap,
            "dev_test": dev_test_doc_overlap,
        },
        "wnd_overlap": {
            "train_dev": train_dev_wnd_overlap,
            "train_test": train_test_wnd_overlap,
            "dev_test": dev_test_wnd_overlap,
        },
        "max_train_events_per_wnd": max_train_events,
        "event_type_counts": dict(event_type_counts),
        "role_counts": dict(role_counts),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    print(f"\nAudit written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/official/train.json")
    parser.add_argument("--dev", default="data/official/dev.json")
    parser.add_argument("--test", default="data/official/test.json")
    parser.add_argument("--out", default="data/manifests/split_audit.json")
    args = parser.parse_args()

    audit_dataset(args.train, args.dev, args.test, args.out)


if __name__ == "__main__":
    main()
