"""Frozen dev rule + single test evaluation for the single-head configuration.

`freeze_rules.py` decodes with the two-head label spaces; the single-head
configuration (`configs/carve_simple.json`) carries one merged 13-type BIO
space, so role spans are decoded from it and filtered to the nine scored roles,
and AAO/trigger spans are decoded from the same tagger and filtered to the AAO
types (the same decoding `ablation_table.py` uses).

Protocol identical to the headline system:
  1. rule = (tau, min_len) maximising MEAN dev Arg-C IoU over seeds 42/13/101,
     on the same grid as `freeze_rules.py`, merge_gap = 0;
  2. that one rule is applied to every seed on test, scored once.

Usage:  python3 scripts/freeze_and_test_simple.py [run_prefix]
"""

import json
import os
import statistics as st
import sys

from transformers import AutoTokenizer

from carve.data import MERGED_LABEL2ID, ROLE_TYPES, load_split
from carve.decode import apply_decoding_rules, posteriors, records_from, spans_with_conf
from carve.evaluate import score, write_jsonl
from carve.paths import repo_root, split_path

MODEL = "microsoft/deberta-v3-large"
SEEDS = [42, 13, 101]
TAUS = [0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]   # = freeze_rules.py
MIN_LENS = [2, 3, 4]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "carve_simple"
ID2MERGED = {i: l for l, i in MERGED_LABEL2ID.items()}
HERE = repo_root()
tok = AutoTokenizer.from_pretrained(MODEL)


def cached(split):
    """Per seed: candidate role spans (with conf), AAO spans, type posteriors."""
    wins = load_split(split)
    out = {}
    for s in SEEDS:
        ck = os.path.join(HERE, "runs", f"{PREFIX}_s{s}", "best.pt")
        rp, _, tp = posteriors([ck], MODEL, wins, tok)
        role = [[x for x in spans_with_conf(rp[i], ID2MERGED) if x[2] in ROLE_TYPES] for i in range(len(wins))]
        aao = [[x for x in spans_with_conf(rp[i], ID2MERGED) if x[2] not in ROLE_TYPES] for i in range(len(wins))]
        out[s] = (role, aao, tp)
    return wins, out


def recs(wins, c, tau, ml):
    role, aao, tp = c
    spans = [apply_decoding_rules(role[i], {"_": tau}, 0, ml, len(wins[i].tokens)) for i in range(len(wins))]
    return records_from(wins, spans, None, tp, aao_precomputed=aao)


dev, cd = cached("dev")
best = None
for ml in MIN_LENS:
    for t in TAUS:
        fs = [score(recs(dev, cd[s], t, ml), split_path("dev"), with_rouge=False)["arg_c_iou"]["f1"] for s in SEEDS]
        if best is None or st.mean(fs) > best["mean"]:
            best = {"mean": st.mean(fs), "std": st.stdev(fs), "tau": t, "min_len": ml, "per_seed": fs}
print(f"FROZEN RULE (dev only): tau={best['tau']} min_len={best['min_len']} merge_gap=0 | "
      f"dev Arg-C {best['mean']:.2f} ± {best['std']:.2f} {[round(f, 2) for f in best['per_seed']]}")
json.dump({"rules": {"tau": {"_": best["tau"]}, "merge_gap": 0, "min_len": best["min_len"]},
           "selection": "max mean dev Arg-C IoU F1 over seeds 42/13/101", **{f"dev_{k}": v for k, v in best.items()}},
          open(os.path.join(HERE, "assets", f"decoding_rules.{PREFIX}.json"), "w"), indent=2)

test, ct = cached("test")
os.makedirs(os.path.join(HERE, "preds"), exist_ok=True)
res = []
for s in SEEDS:
    r = recs(test, ct[s], best["tau"], best["min_len"])
    write_jsonl(r, os.path.join(HERE, "preds", f"test_{PREFIX}_s{s}.jsonl"))
    m = score(r, split_path("test"))
    res.append(m)
    print(f"TEST seed {s}: Arg-C IoU {m['arg_c_iou']['f1']:.2f} (P{m['arg_c_iou']['p']:.1f}/R{m['arg_c_iou']['r']:.1f})"
          f"  Arg-I IoU {m['arg_i_iou']['f1']:.2f}  ROUGE-L {m['trigger_rougeL']['f1']:.2f}")
for k, lab in [("arg_c_iou", "Arg-C IoU"), ("arg_i_iou", "Arg-I IoU"), ("trigger_rougeL", "ROUGE-L")]:
    v = [m[k]["f1"] for m in res]
    print(f"TEST {lab:10s} {st.mean(v):.2f} ± {st.stdev(v):.2f}")
