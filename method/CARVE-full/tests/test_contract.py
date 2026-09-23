"""Correctness tests: data contract, BIO round-trip, and evaluator oracle.

The critical test is `test_oracle_upper_bound`: feeding gold spans through our
own record-building code into the *official* upstream evaluator must produce
~100 on every metric. If it does not, our prediction export is broken and every
downstream number would be meaningless.
"""

import os
import sys


from carve.data import (
    AAO_LABEL2ID,
    ROLE_LABEL2ID,
    bio_to_spans,
    load_split,
    spans_to_bio,
)
from carve.data import default_data_dir
from carve.evaluate import score, to_oneie_record

ID2ROLE = {i: l for l, i in ROLE_LABEL2ID.items()}
ID2AAO = {i: l for l, i in AAO_LABEL2ID.items()}


def test_splits_load():
    for sp, n in [("train", 1278), ("dev", 158), ("test", 163)]:
        w = load_split(sp)
        assert len(w) == n, (sp, len(w), n)
    print("[ok] splits load with expected sizes")


def test_bio_roundtrip():
    """BIO encode->decode must recover every non-conflicting gold span exactly."""
    w = load_split("train")
    recovered = total = 0
    for x in w:
        y = spans_to_bio(x.role_spans, len(x.tokens), ROLE_LABEL2ID)
        got = set(bio_to_spans(y, ID2ROLE))
        for s in x.role_spans:
            total += 1
            recovered += s in got
    rate = recovered / total
    print(f"[ok] BIO round-trip recovers {recovered}/{total} = {rate:.4f} of gold role spans")
    assert rate > 0.97, rate


def test_oracle_upper_bound():
    """Gold spans routed through our exporter + the official evaluator = ceiling."""
    for split in ["dev", "test"]:
        w = load_split(split)
        gold_path = os.path.join(default_data_dir(), f"{split}.oneie.json")
        recs = []
        for x in w:
            trig = next(((s, e) for s, e, t in x.aao_spans if t == "Action"), (0, 1))
            aao = [(s, e, t) for s, e, t in x.aao_spans if t != "Action"]
            recs.append(to_oneie_record(x.sent_id, x.tokens, x.event_type, x.role_spans, aao, trig))
        m = score(recs, gold_path)
        print(
            f"[oracle:{split}] ArgC-IoU {m['arg_c_iou']['f1']:.2f}  ArgI-IoU {m['arg_i_iou']['f1']:.2f}  "
            f"ArgC-EM {m['arg_c_exact']['f1']:.2f}  RougeL {m['trigger_rougeL']['f1']:.2f}"
        )
        assert m["arg_c_iou"]["f1"] > 99.0, m["arg_c_iou"]
        assert m["arg_c_exact"]["f1"] > 99.0, m["arg_c_exact"]


def test_bio_oracle_ceiling():
    """Ceiling actually reachable by a BIO tagger (after conflict resolution)."""
    for split in ["dev"]:
        w = load_split(split)
        gold_path = os.path.join(default_data_dir(), f"{split}.oneie.json")
        recs = []
        for x in w:
            r = bio_to_spans(spans_to_bio(x.role_spans, len(x.tokens), ROLE_LABEL2ID), ID2ROLE)
            a = bio_to_spans(spans_to_bio(x.aao_spans, len(x.tokens), AAO_LABEL2ID), ID2AAO)
            trig = next(((s, e) for s, e, t in a if t == "Action"), (0, 1))
            recs.append(to_oneie_record(x.sent_id, x.tokens, x.event_type, r,
                                        [(s, e, t) for s, e, t in a if t != "Action"], trig))
        m = score(recs, gold_path)
        print(
            f"[bio-ceiling:{split}] ArgC-IoU {m['arg_c_iou']['f1']:.2f}  ArgI-IoU {m['arg_i_iou']['f1']:.2f}  "
            f"ArgC-EM {m['arg_c_exact']['f1']:.2f}  RougeL {m['trigger_rougeL']['f1']:.2f}"
        )
        assert m["arg_c_iou"]["f1"] > 95.0, m["arg_c_iou"]


if __name__ == "__main__":
    test_splits_load()
    test_bio_roundtrip()
    test_oracle_upper_bound()
    test_bio_oracle_ceiling()
    print("\nALL CONTRACT TESTS PASSED")
