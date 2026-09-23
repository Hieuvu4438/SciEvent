"""Decoding-component ablation over the three final verifiers.

Each variant fixes one or more rule components and re-freezes the remaining ones
on dev with the same criterion as the main rule (max MEAN dev Arg-C over the
three verifiers). The frozen variant rule is then applied to the saved test
probabilities (written by `freeze_and_test.py test`). No variant is selected on
test; every row is reported.

Usage:  python3 scripts/ablation.py
"""

import json
import statistics as st

import numpy as np

from hone.data import load_split
from hone.evaluate import score
from hone.paths import split_path
from hone.select import ALPHAS, MIN_LENS, NMS_MODES, THETAS, mix_roles, records, role_given_type
from hone.verifier import build_examples
from scripts.freeze_and_test import LAMBDAS, ROOT, pooled_rows, regroup, run_args

RUNS = ["h2_v42_p3", "h2_v13_p3", "h2_v101_p3"]
PROP = [42, 13, 101]
VARIANTS = [
    ("HONE (full)", {}),
    ("− hybrid role (verifier roles only, α=1)", {"role_alpha": [1.0]}),
    ("− verifier role (proposer roles only, α=0)", {"role_alpha": [0.0]}),
    ("− joint type (λ=0)", {"type_lambda": [0.0]}),
    ("− hybrid role − joint type (pure verifier)", {"role_alpha": [1.0], "type_lambda": [0.0]}),
    ("− collision rule 'any' (iou only)", {"nms": ["iou"]}),
]


def load(split):
    by = {w.sent_id: w for w in load_split(split)}
    rows, wins = pooled_rows(split, PROP, by, run_args(RUNS))
    ex = build_examples(by, rows, with_labels=False)
    name = "dev_probs.npy" if split == "dev" else "test_probs.npy"
    return rows, wins, [regroup(ex, np.load(ROOT / "runs" / r / name), rows) for r in RUNS]


def metrics(split_data, rule, lrt, gold, keys=("arg_c_iou",), rouge=False):
    rows, wins, per_run = split_data
    out = []
    for pw in per_run:
        m = score(records(wins, rows, mix_roles(rows, pw, rule["role_alpha"]), rule["theta"],
                          rule["min_len"], rule["nms"], rule["type_lambda"], lrt), gold, with_rouge=rouge)
        out.append({k: m[k]["f1"] for k in keys})
    return out


def main():
    lrt = role_given_type(load_split("train"))
    dev, test = load("dev"), load("test")
    g_dev, g_test = split_path("dev"), split_path("test")
    table = []
    for name, fixed in VARIANTS:
        grid = {"role_alpha": ALPHAS, "type_lambda": LAMBDAS, "nms": NMS_MODES,
                "min_len": MIN_LENS, "theta": THETAS} | fixed
        best = None
        for a in grid["role_alpha"]:
            for lam in grid["type_lambda"]:
                for nms in grid["nms"]:
                    for ml in grid["min_len"]:
                        for th in grid["theta"]:
                            rule = {"role_alpha": a, "type_lambda": lam, "nms": nms, "min_len": ml, "theta": th}
                            mu = st.mean(x["arg_c_iou"] for x in metrics(dev, rule, lrt, g_dev))
                            if best is None or mu > best[0]:
                                best = (mu, rule)
        d = metrics(dev, best[1], lrt, g_dev)
        t = metrics(test, best[1], lrt, g_test, keys=("arg_c_iou", "arg_i_iou"))
        row = {"variant": name, "rule": best[1],
               "dev_arg_c": [x["arg_c_iou"] for x in d],
               "test_arg_c": [x["arg_c_iou"] for x in t], "test_arg_i": [x["arg_i_iou"] for x in t]}
        table.append(row)
        f = lambda v: f"{st.mean(v):.2f} ± {st.stdev(v):.2f}"
        print(f"{name:46s} dev C {f(row['dev_arg_c'])} | test C {f(row['test_arg_c'])}  "
              f"I {f(row['test_arg_i'])} | {best[1]}", flush=True)
    json.dump(table, open(ROOT / "assets" / "ablation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
