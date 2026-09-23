"""Bootstrap confidence intervals for the headline test metrics.

We report mean +- std over three seeds, which quantifies training variance but
not sampling variance: the test split has only 163 windows, so a reviewer is
right to ask whether an 8.87-point margin could be an artifact of which 163
windows happened to be sampled.

This script resamples test *windows* with replacement (the unit of annotation),
recomputes the official metrics on each resample, and reports percentile
intervals. Matching uses the benchmark's own functions, so the criterion is
identical to the headline numbers.

A paired significance test against OneIE is not possible: the SciEvent paper
reports aggregate numbers only, and per-window OneIE predictions are not
released. We therefore report our own interval and compare its lower bound to
the published baseline, which is the strongest claim the available evidence
supports.
"""

import collections
import glob
import json
import os
import statistics as st
import sys

import numpy as np

from carve.data import EXCLUDED_FROM_SCORING
from carve.evaluate import official_scorer
from carve.paths import repo_root, split_path

N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
SEED = 0

BASELINE = {"arg_c_iou": 41.61, "arg_i_iou": 53.57, "trigger_rougeL": 75.08}

M = official_scorer()
gold_data = M.load_jsonl(split_path(SPLIT))
tokens = {e["sent_id"]: e["tokens"] for e in gold_data}
_, gold_roles = M.extract_triggers_and_roles(gold_data)
sent_ids = [e["sent_id"] for e in gold_data]


def counts_per_window(pred_roles, match_fn, use_role):
    """(matched, n_pred, n_gold) per window, so a bootstrap can just sum them."""
    exclude = EXCLUDED_FROM_SCORING
    out = {}
    for sid in sent_ids:
        g = [x for x in gold_roles[sid] if x[1][2] not in exclude]
        p = [x for x in pred_roles.get(sid, []) if x[1][2] not in exclude]
        matched, used = 0, set()
        for pr in p:
            for i, gr in enumerate(g):
                if i in used or pr[0][2] != gr[0][2]:
                    continue
                if use_role and pr[1][2] != gr[1][2]:
                    continue
                if match_fn(pr[1][:2], gr[1][:2]):
                    matched += 1
                    used.add(i)
                    break
        out[sid] = (matched, len(p), len(g))
    return out


def f1_from(counts, idx):
    m = sum(counts[s][0] for s in idx)
    p = sum(counts[s][1] for s in idx)
    g = sum(counts[s][2] for s in idx)
    pr = m / p if p else 0.0
    rc = m / g if g else 0.0
    return 2 * pr * rc / (pr + rc) * 100 if pr + rc else 0.0


def rouge_per_window(pred_roles):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    out = {}
    for sid in sent_ids:
        g = " ".join(M.extract_summary_tuple(gold_roles[sid], tokens[sid], sid)[0]).strip()
        p = " ".join(M.extract_summary_tuple(pred_roles.get(sid, []), tokens[sid], sid)[0]).strip()
        out[sid] = sc.score(g, p)["rougeL"].fmeasure * 100 if (g or p) else None
    return out


files = sorted(glob.glob(os.path.join(repo_root(), "preds", f"{SPLIT}_s*.jsonl")))
assert files, f"no predictions found in preds/{SPLIT}_s*.jsonl"
print(f"bootstrapping {N_BOOT} resamples over {len(sent_ids)} {SPLIT} windows, "
      f"{len(files)} seeds\n")

rng = np.random.default_rng(SEED)
boot_idx = [[sent_ids[i] for i in rng.integers(0, len(sent_ids), len(sent_ids))]
            for _ in range(N_BOOT)]

results = collections.defaultdict(list)
for f in files:
    preds = M.load_jsonl(f)
    _, pred_roles = M.extract_triggers_and_roles(preds)
    c_c = counts_per_window(pred_roles, M.iou_overlap, use_role=True)
    c_i = counts_per_window(pred_roles, M.iou_overlap, use_role=False)
    rg = rouge_per_window(pred_roles)
    results["arg_c_iou"].append([f1_from(c_c, idx) for idx in boot_idx])
    results["arg_i_iou"].append([f1_from(c_i, idx) for idx in boot_idx])
    results["trigger_rougeL"].append(
        [st.mean([rg[s] for s in idx if rg[s] is not None]) for idx in boot_idx])

print(f"{'metric':18} {'point':>7} {'95% CI':>18} {'baseline':>9} {'CI low - base':>14}")
print("-" * 72)
for k in ["arg_c_iou", "arg_i_iou", "trigger_rougeL"]:
    # average the seeds inside each bootstrap replicate, so the interval covers
    # sampling variance for the system we actually report (the 3-seed mean)
    per_rep = np.mean(np.array(results[k]), axis=0)
    lo, hi = np.percentile(per_rep, [2.5, 97.5])
    point = float(np.mean(per_rep))
    b = BASELINE[k]
    flag = "" if lo > b else "   <-- CI includes baseline"
    print(f"{k:18} {point:7.2f} [{lo:6.2f}, {hi:6.2f}] {b:9.2f} {lo - b:+14.2f}{flag}")

print(f"\nFraction of resamples where the 3-seed mean beats the published baseline:")
for k in ["arg_c_iou", "arg_i_iou", "trigger_rougeL"]:
    per_rep = np.mean(np.array(results[k]), axis=0)
    print(f"  {k:18} {float((per_rep > BASELINE[k]).mean()) * 100:6.2f}%")
