"""Control: the same pooled 3-seed candidates, decoded WITHOUT the verifier.

Isolates the verifier's contribution from candidate pooling. Each candidate's
keep score comes from proposer evidence only, role from its (normalised) role
mass; decoding uses the same collision-aware decoder and grid as HONE. Three
proposer-only keep scores are compared on dev:

    conf        mean token posterior, averaged over the seeds that proposed it
                (CARVE's own score)
    conf_max    max over those seeds
    agree_conf  conf * n_agree / 3   (seed agreement as extra evidence)

The best (score, rule) on dev is frozen and applied once to test.
Usage:  python3 scripts/pooled_threshold_control.py
"""

import numpy as np

from hone import candidates as C
from hone.data import load_split
from hone.evaluate import score
from hone.paths import split_path
from hone.select import MIN_LENS, NMS_MODES, THETAS, records

PROP = [42, 13, 101]
SCORES = {
    "conf": lambda c: c["conf"],
    "conf_max": lambda c: c.get("conf_max", c["conf"]),
    "agree_conf": lambda c: c["conf"] * c.get("n_agree", 1) / 3.0,
}


def rows_for(split):
    by = {w.sent_id: w for w in load_split(split)}
    rows = C.pool_rows([C.load(f"data/cands/{split}_s{s}.jsonl") for s in PROP])
    return rows, [by[r["sent_id"]] for r in rows]


def probs(rows, keyfn):
    out = []
    for r in rows:
        p = np.zeros((len(r["cands"]), len(C.LABELS)), dtype=np.float32)
        for j, c in enumerate(r["cands"]):
            k = keyfn(c)
            rm = np.asarray(c["role_mass"], dtype=float)
            p[j, 0] = 1 - k
            p[j, 1:] = k * rm / max(1e-9, rm.sum())
        out.append(p)
    return out


def main():
    dev, test = rows_for("dev"), rows_for("test")
    best = None
    for name, fn in SCORES.items():
        pw = probs(dev[0], fn)
        b = None
        for nms in NMS_MODES:
            for ml in MIN_LENS:
                for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95]:
                    f = score(records(dev[1], dev[0], pw, th, ml, nms), split_path("dev"),
                              with_rouge=False)["arg_c_iou"]["f1"]
                    if b is None or f > b[0]:
                        b = (f, name, th, ml, nms)
        print(f"dev  {name:10s} Arg-C {b[0]:.2f}  (θ {b[2]}, ml {b[3]}, {b[4]})")
        if best is None or b[0] > best[0]:
            best = b
    f, name, th, ml, nms = best
    m = score(records(test[1], test[0], probs(test[0], SCORES[name]), th, ml, nms), split_path("test"),
              with_rouge=False)
    print(f"\nfrozen on dev: {name}, θ {th}, ml {ml}, {nms}  ->  TEST Arg-C {m['arg_c_iou']['f1']:.2f}"
          f"  Arg-I {m['arg_i_iou']['f1']:.2f}")


if __name__ == "__main__":
    main()
