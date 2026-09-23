"""Paired bootstrap on test: HONE vs CARVE (and vs the paper's best baseline).

Resamples test windows with replacement (the unit of annotation). In every
replicate the *same* windows are scored for both systems, each system's F1 is
the mean over its three runs, and the difference HONE - CARVE is recorded. A
paired interval is available here because CARVE's per-window test predictions
are our own (frozen, from method/SciEvent-Next); for OneIE only the published
aggregate exists, so that comparison is the CI lower bound vs the number.

Matching uses the benchmark's functions (IoU > 0.5, greedy, event type must
agree, Agent/PrimaryObject/SecondaryObject excluded), identical to headline
numbers. Runs after the single frozen test evaluation; reads predictions only.

Usage:  python3 scripts/bootstrap_ci.py 'preds/test_h2_v*_p3.jsonl' [N_BOOT] [REF_GLOB]
REF_GLOB (optional, relative to this repo) replaces CARVE's 3 seeds as the
reference system, e.g. to compare two single-seed systems.
"""

import glob
import sys

import numpy as np

from hone.data import EXCLUDED_FROM_SCORING
from hone.evaluate import official_scorer
from hone.paths import repo_root, split_path

ROOT = repo_root()
HONE_GLOB = sys.argv[1] if len(sys.argv) > 1 else "preds/test_h2_v*_p3.jsonl"
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
CARVE_GLOB = (str(ROOT / sys.argv[3]) if len(sys.argv) > 3
              else str(ROOT.parent / "SciEvent-Next" / "artifacts" / "preds" / "test_preds_s*.jsonl"))
BASELINE = {"arg_c_iou": 41.61, "arg_i_iou": 53.57}   # OneIE, SciEvent paper Table 3/4

M = official_scorer()
gold = M.load_jsonl(split_path("test"))
sent_ids = [e["sent_id"] for e in gold]
_, gold_roles = M.extract_triggers_and_roles(gold)


def counts(pred_file, use_role):
    _, pr = M.extract_triggers_and_roles(M.load_jsonl(pred_file))
    out = np.zeros((len(sent_ids), 3))
    for k, sid in enumerate(sent_ids):
        g = [x for x in gold_roles[sid] if x[1][2] not in EXCLUDED_FROM_SCORING]
        p = [x for x in pr.get(sid, []) if x[1][2] not in EXCLUDED_FROM_SCORING]
        m, used = 0, set()
        for a in p:
            for i, b in enumerate(g):
                if i in used or a[0][2] != b[0][2] or (use_role and a[1][2] != b[1][2]):
                    continue
                if M.iou_overlap(a[1][:2], b[1][:2]):
                    m += 1
                    used.add(i)
                    break
        out[k] = (m, len(p), len(g))
    return out


def f1(c):   # c: [..., 3] summed counts
    p = np.where(c[..., 1] > 0, c[..., 0] / np.maximum(c[..., 1], 1), 0)
    r = np.where(c[..., 2] > 0, c[..., 0] / np.maximum(c[..., 2], 1), 0)
    return np.where(p + r > 0, 200 * p * r / np.maximum(p + r, 1e-12), 0)


hone_files = sorted(glob.glob(str(ROOT / HONE_GLOB)))
carve_files = sorted(glob.glob(CARVE_GLOB))
assert len(hone_files) >= 1 and len(carve_files) >= 1, (hone_files, carve_files)
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, len(sent_ids), len(sent_ids)), minlength=len(sent_ids))
              for _ in range(N_BOOT)])                          # [B, n_windows] multiplicities
print(f"{N_BOOT} paired resamples of {len(sent_ids)} test windows; "
      f"system runs {len(hone_files)}, reference runs {len(carve_files)}\n")
print(f"{'metric':10} {'sys':>7} {'ref':>7} {'Δ':>6} {'Δ 95% CI':>17} {'P(Δ>0)':>7} "
      f"{'sys CI':>17} {'OneIE':>6}")
for key, use_role in [("arg_c_iou", True), ("arg_i_iou", False)]:
    def sys_f1(files):
        cs = [counts(f, use_role) for f in files]
        point = np.mean([f1(c.sum(0)) for c in cs])
        boot = np.mean([f1(W @ c) for c in cs], axis=0)
        return point, boot
    hp, hb = sys_f1(hone_files)
    cp, cb = sys_f1(carve_files)
    d = hb - cb
    lo, hi = np.percentile(d, [2.5, 97.5])
    hlo, hhi = np.percentile(hb, [2.5, 97.5])
    print(f"{key:10} {hp:7.2f} {cp:7.2f} {hp - cp:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] {np.mean(d > 0):7.3f} "
          f"[{hlo:6.2f}, {hhi:6.2f}] {BASELINE[key]:6.2f}")
