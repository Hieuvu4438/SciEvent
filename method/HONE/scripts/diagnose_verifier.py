"""Does the verifier add information beyond the proposer?

On dev candidates, compares
  * keep decision: ROC-AUC of the verifier's keep score (1 - p_reject) against
    CARVE's own score (mean token posterior), for "candidate matches some gold";
  * role decision: accuracy of the verifier's argmax role vs the proposer's
    role, on candidates that do match gold;
  * hybrid decoding: keep score from the verifier, role from an alpha-mix of
    verifier role probabilities and the proposer's role-mass vector.

Usage:  python3 scripts/diagnose_verifier.py RUN_NAME --prop_seeds 42
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from hone import candidates as C
from hone.data import ROLE_TYPES, load_split
from hone.evaluate import score
from hone.paths import repo_root, split_path
from hone.select import MIN_LENS, NMS_MODES, THETAS, records
from hone.verifier import build_examples

ROOT = repo_root()
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--prop_seeds", type=int, nargs="+", default=[42])
a = ap.parse_args()

dev = load_split("dev")
by = {w.sent_id: w for w in dev}
rows = C.pool_rows([C.load(str(ROOT / "data" / "cands" / f"dev_s{s}.jsonl")) for s in a.prop_seeds])
import json
args = json.load(open(ROOT / "runs" / a.run / "log.json"))["args"]
for r in rows:
    if args.get("compose"):
        r["cands"] = C.compose(r["cands"], args["compose_run"], args["compose_gap"])
    C.label(by[r["sent_id"]], r["cands"])
wins = [by[r["sent_id"]] for r in rows]
ex = build_examples(by, rows)
probs = np.load(ROOT / "runs" / a.run / "dev_probs.npy")

flat = [c for r in rows for c in r["cands"]]
is_pos = np.array([c["label"] != "reject" for c in flat])
pos_right_role = np.array([c["label"] != "reject" and c["label"] == c["role"] for c in flat])
keep_v = 1.0 - probs[:, 0]
conf_p = np.array([c["conf"] for c in flat])
print(f"dev candidates {len(flat)}  positive {is_pos.mean():.1%}")
print(f"\nKEEP decision ROC-AUC (target: candidate IoU-matches a gold span)")
print(f"  proposer confidence (CARVE's score) : {roc_auc_score(is_pos, conf_p):.4f}")
print(f"  verifier keep score                 : {roc_auc_score(is_pos, keep_v):.4f}")
print(f"KEEP-with-correct-role ROC-AUC (target: matches gold AND proposer role is right)")
print(f"  proposer confidence                 : {roc_auc_score(pos_right_role, conf_p):.4f}")
print(f"  verifier p(proposed role)           : "
      f"{roc_auc_score(pos_right_role, [probs[i, C.LABEL2ID[c['role']]] for i, c in enumerate(flat)]):.4f}")

pos_idx = np.where(is_pos)[0]
v_role = [C.LABELS[1 + int(np.argmax(probs[i, 1:]))] for i in pos_idx]
p_role = [flat[i]["role"] for i in pos_idx]
gold = [flat[i]["label"] for i in pos_idx]
print(f"\nROLE accuracy on {len(pos_idx)} gold-matched candidates")
print(f"  proposer role : {np.mean([a == b for a, b in zip(p_role, gold)]):.1%}")
print(f"  verifier role : {np.mean([a == b for a, b in zip(v_role, gold)]):.1%}")

# hybrid: verifier keep score, role from alpha-mixed distributions
print("\nHYBRID decoding (dev-tuned rule per alpha)")
off = 0
per_w_base = []
for r in rows:
    per_w_base.append(probs[off:off + len(r["cands"])])
    off += len(r["cands"])
g = split_path("dev")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    mixed = []
    for r, pw in zip(rows, per_w_base):
        m = pw.copy()
        for j, c in enumerate(r["cands"]):
            pm = np.asarray(c["role_mass"]) / max(1e-9, sum(c["role_mass"]))
            vr = pw[j, 1:] / max(1e-9, pw[j, 1:].sum())
            m[j, 1:] = (1 - pw[j, 0]) * (alpha * vr + (1 - alpha) * pm)
        mixed.append(m)
    best = max(((score(records(wins, rows, mixed, th, ml, nms), g, with_rouge=False)["arg_c_iou"]["f1"], th, ml, nms)
                for nms in NMS_MODES for ml in MIN_LENS for th in THETAS))
    print(f"  alpha={alpha:4.2f} (0 = proposer role, 1 = verifier role): Arg-C {best[0]:.2f}  "
          f"[th={best[1]} ml={best[2]} nms={best[3]}]")
