"""Freeze the decoding rule on dev, without letting it fit one particular seed.

H7 (posterior ensembling) was falsified: averaging posteriors across seeds helps
argmax decoding but *hurts* calibrated decoding, because averaging compresses the
confidence distribution the threshold operates on. The method is therefore a
single model, and the honest report is the mean +- std over seeds.

To avoid tuning the threshold to whichever seed happens to look best on dev, the
rule is chosen to maximise the **mean dev Arg-C IoU F1 across all seeds**, and
that single rule is then frozen for the test evaluation of every seed.
"""

import json
import os
import statistics
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from scievent_next.data import default_data_dir, load_split  # noqa: E402
from scievent_next.evaluate import score  # noqa: E402
from scievent_next.infer import build_records, posteriors  # noqa: E402

MODEL = "microsoft/deberta-v3-large"
SEEDS = [42, 13, 101]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# optional: scripts/freeze_rules.py <run_prefix> <data_dir> <out_name>
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "final"
DATA_DIR = sys.argv[2] if len(sys.argv) > 2 else None
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "decoding_rules.json"
CKPT = {s: os.path.join(HERE, "artifacts", "runs", f"{PREFIX}_s{s}", "best.pt") for s in SEEDS}

TAUS = [0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
MIN_LENS = [2, 3, 4]

tok = AutoTokenizer.from_pretrained(MODEL)
dev = load_split("dev", DATA_DIR)
gold = os.path.join(DATA_DIR or default_data_dir(), "dev.oneie.json")

post = {s: posteriors([CKPT[s]], MODEL, dev, tok) for s in SEEDS}

best = None
print(f"{'tau':>5} {'ml':>3} | " + " ".join(f"s{s:<5}" for s in SEEDS) + " |   mean     std")
for ml in MIN_LENS:
    for t in TAUS:
        fs = []
        for s in SEEDS:
            rp, ap, tp = post[s]
            fs.append(score(build_records(dev, rp, ap, tp, {"_": t}, 0, ml), gold)["arg_c_iou"]["f1"])
        mu, sd = statistics.mean(fs), statistics.stdev(fs)
        print(f"{t:5.2f} {ml:3d} | " + " ".join(f"{f:6.2f}" for f in fs) + f" | {mu:6.2f}  {sd:6.2f}")
        if best is None or mu > best["mean"]:
            best = {"mean": mu, "std": sd, "tau": t, "min_len": ml, "per_seed": fs}

print(f"\nFROZEN RULE: tau={best['tau']}  min_len={best['min_len']}  merge_gap=0")
print(f"  dev Arg-C IoU F1 across seeds: {best['mean']:.2f} +- {best['std']:.2f}  {best['per_seed']}")

out = os.path.join(HERE, "artifacts", OUT_NAME)
json.dump({"rules": {"tau": {"_": best["tau"]}, "merge_gap": 0, "min_len": best["min_len"]},
           "selection": "max mean dev Arg-C IoU F1 over seeds 42/13/101 (single-model decoding)",
           "dev_arg_c_iou_f1_mean": best["mean"], "dev_arg_c_iou_f1_std": best["std"],
           "dev_per_seed": best["per_seed"], "seeds": SEEDS},
          open(out, "w"), indent=2)
print(f"wrote {out}")
