"""Ablation table: every row differs from the full model by exactly ONE factor.

Each configuration is trained on the same three seeds and re-scored on dev with
the same decoding grid, so the deltas are like-for-like. The single-head
configuration carries a merged 13-type label space, which this script decodes
with the matching inventory before filtering to the nine scored roles.
"""

import os
import statistics as st

import numpy as np
from transformers import AutoTokenizer

from carve.data import (
    AAO_LABEL2ID,
    MERGED_LABEL2ID,
    ROLE_LABEL2ID,
    ROLE_TYPES,
    load_split,
)
from carve.decode import _detect_heads, apply_decoding_rules, posteriors, records_from, spans_with_conf
from carve.evaluate import score
from carve.paths import repo_root, split_path

MODEL = "microsoft/deberta-v3-large"
SEEDS = [42, 13, 101]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.90, 0.92, 0.95]
MIN_LENS = [2, 3, 4]

ID2ROLE = {i: l for l, i in ROLE_LABEL2ID.items()}
ID2AAO = {i: l for l, i in AAO_LABEL2ID.items()}
ID2MERGED = {i: l for l, i in MERGED_LABEL2ID.items()}

ROWS = [
    ("Full model (CARVE)", "carve"),
    ("  − event-type conditioning", "abl_no_type_cond"),
    ("  − two disjoint heads", "abl_single_head"),
    ("  + linear-chain CRF", "abl_crf"),
    ("  + span-level role head", "abl_span_role"),
    ("  + layer-wise LR decay", "abl_llrd"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
dev = load_split("dev")
gold = split_path("dev")


def best_dev(prefix):
    """Per seed, the best dev Arg-C IoU over the shared decoding grid."""
    out = []
    for s in SEEDS:
        run_dir = os.path.join(repo_root(), "runs", f"{prefix}_s{s}")
        ck = os.path.join(run_dir, "best.pt")
        # log.json is written only after the epoch loop finishes, so it is the
        # signal that best.pt is complete rather than mid-write.
        if not os.path.exists(os.path.join(run_dir, "log.json")):
            return None
        heads = _detect_heads(ck)
        rp, ap, tp = posteriors([ck], MODEL, dev, tok, use_crf=heads["use_crf"])
        single = heads["single_head"]
        id2role = ID2MERGED if single else ID2ROLE
        id2aao = ID2MERGED if single else ID2AAO

        # cache the confidence-annotated candidate spans once per seed
        role_cand, aao_spans = [], []
        for i in range(len(dev)):
            r = [x for x in spans_with_conf(rp[i], id2role) if x[2] in ROLE_TYPES]
            a = [x for x in spans_with_conf(ap[i], id2aao) if x[2] not in ROLE_TYPES]
            role_cand.append(r)
            aao_spans.append(a)

        best = 0.0
        for ml in MIN_LENS:
            for t in TAUS:
                spans = [apply_decoding_rules(role_cand[i], {"_": t}, 0, ml, len(dev[i].tokens))
                         for i in range(len(dev))]
                recs = records_from(dev, spans, None, tp, aao_precomputed=aao_spans)
                best = max(best, score(recs, gold, with_rouge=False)["arg_c_iou"]["f1"])
        out.append(best)
    return out


full = None
print(f"{'configuration':34} {'dev Arg-C IoU F1':>18} {'Δ':>8}")
print("-" * 64)
for label, prefix in ROWS:
    vals = best_dev(prefix)
    if vals is None:
        print(f"{label:34} {'(not trained)':>18}")
        continue
    mu, sd = st.mean(vals), st.stdev(vals)
    if full is None:
        full = mu
        print(f"{label:34} {mu:11.2f} ± {sd:4.2f} {'—':>8}")
    else:
        print(f"{label:34} {mu:11.2f} ± {sd:4.2f} {mu - full:+8.2f}")
print("-" * 64)
print(f"seeds: {SEEDS}   decoding grid: tau {TAUS}, min_len {MIN_LENS}")
