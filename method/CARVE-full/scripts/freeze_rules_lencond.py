"""Freeze a length-conditioned decoding rule on dev, seed-agnostically.

The shipped rule drops any span shorter than `min_len` regardless of confidence.
That is a length-conditioned threshold whose short-span branch is infinite, and
it is the direct cause of the 14.3% recall on 1-4 token gold spans.

This script searches the same rule family with a *finite* short-span threshold:

    tau_eff(len) = tau_short   if len <  short_len
                 = tau_long    if len >= short_len

Parameter count is three, identical to the shipped rule (tau, min_len,
merge_gap=0), so this does not add dev-fitted capacity.

Selection criterion is unchanged: maximise the MEAN dev Arg-C IoU F1 across
seeds, never a single seed.

Usage:  python3 scripts/freeze_rules_lencond.py [run_prefix] [data_dir] [out_name]
"""

import json
import os
import statistics
import sys

from transformers import AutoTokenizer

from carve.data import load_split
from carve.decode import build_records, posteriors
from carve.evaluate import score
from carve.paths import default_data_dir, repo_root, split_path

MODEL = "microsoft/deberta-v3-large"
SEEDS = [42, 13, 101]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "carve"
DATA_DIR = sys.argv[2] if len(sys.argv) > 2 else None
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "decoding_rules_lencond.json"

CKPT = {s: os.path.join(repo_root(), "runs", f"{PREFIX}_s{s}", "best.pt") for s in SEEDS}

TAU_LONG = [0.80, 0.85, 0.88, 0.90, 0.92]
TAU_SHORT = [0.90, 0.93, 0.95, 0.97, 0.99]
SHORT_LEN = [3, 4, 5]

tok = AutoTokenizer.from_pretrained(MODEL)
dev = load_split("dev", DATA_DIR)
gold = split_path("dev", DATA_DIR)

post = {s: posteriors([CKPT[s]], MODEL, dev, tok) for s in SEEDS}


def mean_dev(tau_long, tau_short, short_len):
    fs = []
    for s in SEEDS:
        rp, ap, tp = post[s]
        recs = build_records(dev, rp, ap, tp, {"_": tau_long}, 0, 1,
                             tau_short=tau_short, short_len=short_len)
        fs.append(score(recs, gold)["arg_c_iou"]["f1"])
    return statistics.mean(fs), statistics.stdev(fs), fs


# Reference point: the shipped rule (infinite short-span threshold, min_len=3).
ref = []
for s in SEEDS:
    rp, ap, tp = post[s]
    ref.append(score(build_records(dev, rp, ap, tp, {"_": 0.90}, 0, 3), gold)["arg_c_iou"]["f1"])
print(f"shipped rule (tau=0.90, min_len=3): dev Arg-C IoU "
      f"{statistics.mean(ref):.2f} +- {statistics.stdev(ref):.2f}\n")

best = None
print(f"{'tau_long':>8} {'tau_short':>9} {'short_len':>9} | " +
      " ".join(f"s{s:<5}" for s in SEEDS) + " |   mean     std")
for sl in SHORT_LEN:
    for tl in TAU_LONG:
        for ts in TAU_SHORT:
            if ts < tl:
                continue  # a short span must never be easier to keep than a long one
            mu, sd, fs = mean_dev(tl, ts, sl)
            print(f"{tl:8.2f} {ts:9.2f} {sl:9d} | " + " ".join(f"{f:6.2f}" for f in fs) +
                  f" | {mu:6.2f}  {sd:6.2f}")
            if best is None or mu > best["mean"]:
                best = {"mean": mu, "std": sd, "tau_long": tl, "tau_short": ts,
                        "short_len": sl, "per_seed": fs}

print(f"\nFROZEN LENGTH-CONDITIONED RULE: tau_long={best['tau_long']} "
      f"tau_short={best['tau_short']} short_len={best['short_len']}  merge_gap=0")
print(f"  dev Arg-C IoU F1 across seeds: {best['mean']:.2f} +- {best['std']:.2f}  {best['per_seed']}")
print(f"  vs shipped rule: {best['mean'] - statistics.mean(ref):+.2f}")

out = os.path.join(repo_root(), "assets", OUT_NAME)
json.dump({"rules": {"tau": {"_": best["tau_long"]}, "merge_gap": 0, "min_len": 1,
                     "tau_short": best["tau_short"], "short_len": best["short_len"]},
           "selection": "max mean dev Arg-C IoU F1 over seeds 42/13/101 (single-model decoding)",
           "dev_arg_c_iou_f1_mean": best["mean"], "dev_arg_c_iou_f1_std": best["std"],
           "dev_per_seed": best["per_seed"],
           "shipped_rule_dev_mean": statistics.mean(ref), "seeds": SEEDS},
          open(out, "w"), indent=2)
print(f"wrote {out}")
