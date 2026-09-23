"""Controlled test of the geometry hypothesis, within a single benchmark.

The central claim of this work is that a clause-level span tagger is the right
model family when targets are clause-sized, and the wrong one when they are
mention-sized. Testing that normally requires a second benchmark, which changes
the data, the label inventory, the evaluator and the model all at once.

SciEvent allows a stricter test. The *same windows* contain two groups of target
spans, extracted by the *same model* in the *same forward pass*:

  * the 9 scored semantic roles  -- clause-sized (mean 10.2-15.5 tokens)
  * Agent / Primary / SecondaryObject -- mention-sized (mean 2.4 / 5.9 / 6.4)

Everything else is held constant: encoder, training data, optimisation, decoding.
Only the geometry of the target differs.

Matching uses the benchmark's own `iou_overlap` function, so the criterion is
identical to the headline metric. The aggregation is ours, because the benchmark
deliberately excludes these three roles from Arg-I/Arg-C -- we therefore report
this as an analysis, never as a benchmark number.
"""

import collections
import statistics as st
import sys

from transformers import AutoTokenizer

from carve.data import AAO_TYPES, ROLE_TYPES, load_split
from carve.decode import ID2AAO, ID2ROLE, posteriors, spans_with_conf
from carve.evaluate import official_scorer

MODEL = "microsoft/deberta-v3-large"
SEEDS = [42, 13, 101]
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
TAU, MIN_LEN = 0.90, 3

iou_overlap = official_scorer().iou_overlap  # the benchmark's own matcher


def prf(matched, pred, gold):
    p = matched / pred if pred else 0.0
    r = matched / gold if gold else 0.0
    return p * 100, r * 100, (2 * p * r / (p + r) * 100 if p + r else 0.0)


def score_group(windows, preds_per_window, gold_attr, types):
    """Greedy one-to-one IoU matching per window, per role type, as the official
    scorer does for the roles it covers."""
    stats = collections.defaultdict(lambda: [0, 0, 0])  # matched, pred, gold
    for w, pred in zip(windows, preds_per_window):
        gold = [g for g in getattr(w, gold_attr) if g[2] in types]
        pr = [p for p in pred if p[2] in types]
        for t in types:
            g = [x for x in gold if x[2] == t]
            p = [x for x in pr if x[2] == t]
            stats[t][1] += len(p)
            stats[t][2] += len(g)
            used = set()
            for ps, pe, _ in p:
                for i, (gs, ge, _) in enumerate(g):
                    if i in used:
                        continue
                    if iou_overlap((ps, pe), (gs, ge)):
                        stats[t][0] += 1
                        used.add(i)
                        break
    return stats


tok = AutoTokenizer.from_pretrained(MODEL)
windows = load_split(SPLIT)

mean_len = collections.defaultdict(list)
for w in windows:
    for s, e, t in list(w.role_spans) + list(w.aao_spans):
        mean_len[t].append(e - s)

per_seed = []
for seed in SEEDS:
    rp, ap, tp = posteriors([f"runs/carve_s{seed}/best.pt"], MODEL, windows, tok)
    # the scored-role head, decoded exactly as the shipped system does
    role_pred = [[(s, e, t) for s, e, t, c in spans_with_conf(rp[i], ID2ROLE)
                  if c >= TAU and (e - s) >= MIN_LEN] for i in range(len(windows))]
    # the AAO head, decoded at argmax (the shipped system never thresholds it)
    aao_pred = [[(s, e, t) for s, e, t, _ in spans_with_conf(ap[i], ID2AAO)]
                for i in range(len(windows))]
    per_seed.append((score_group(windows, role_pred, "role_spans", ROLE_TYPES),
                     score_group(windows, aao_pred, "aao_spans", [t for t in AAO_TYPES if t != "Action"])))

print(f"Controlled geometry test on {SPLIT}, mean over {len(SEEDS)} seeds.")
print("Same model, same forward pass, same IoU>0.5 matcher. Only target geometry differs.\n")
print(f"{'target':18} {'group':12} {'mean len':>9} {'gold':>6} {'P':>7} {'R':>7} {'F1':>7}")
print("-" * 72)

rows = []
for group_idx, (group_name, types) in enumerate(
        [("clause-sized", ROLE_TYPES), ("mention-sized", [t for t in AAO_TYPES if t != "Action"])]):
    for t in types:
        vals = []
        for s in per_seed:
            m, p, g = s[group_idx][t]
            vals.append(prf(m, p, g))
        if not mean_len[t]:
            continue
        gold_n = per_seed[0][group_idx][t][2]
        if gold_n < 5:
            continue  # too few to be meaningful
        f1 = st.mean(v[2] for v in vals)
        rows.append((t, group_name, st.mean(mean_len[t]), gold_n,
                     st.mean(v[0] for v in vals), st.mean(v[1] for v in vals), f1))

for t, g, ml, n, p, r, f in sorted(rows, key=lambda x: -x[2]):
    print(f"{t:18} {g:12} {ml:9.1f} {n:6d} {p:7.2f} {r:7.2f} {f:7.2f}")

print("-" * 72)
for group_name in ["clause-sized", "mention-sized"]:
    sel = [x for x in rows if x[1] == group_name]
    tot_gold = sum(x[3] for x in sel)
    micro = sum(x[5] * x[3] for x in sel) / tot_gold  # gold-weighted recall
    print(f"{group_name:12}  n_gold={tot_gold:4d}  gold-weighted F1 = "
          f"{sum(x[6] * x[3] for x in sel) / tot_gold:6.2f}   recall = {micro:6.2f}")
