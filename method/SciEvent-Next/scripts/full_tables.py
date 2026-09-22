"""Aggregate every official metric over the frozen test predictions (3 seeds)
and print the comparison against the paper's Table 3 and Table 4.

Usage:
    python3 scripts/full_tables.py [split] [pred_glob] [data_dir]
"""

import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scievent_next.data import default_data_dir  # noqa: E402
from scievent_next.evaluate import prf, score  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
GLOB = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "artifacts", "preds", "test_preds_s*.jsonl")
DATA_DIR = sys.argv[3] if len(sys.argv) > 3 else default_data_dir()
GOLD = os.path.join(DATA_DIR, f"{SPLIT}.oneie.json")

files = sorted(glob.glob(GLOB))
assert files, f"no predictions matched {GLOB}"
mets = []
for f in files:
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    mets.append(score(recs, GOLD))
print(f"aggregating {len(files)} seeds over {SPLIT}: {[os.path.basename(f) for f in files]}\n")


def agg(fn):
    vals = [fn(m) for m in mets]
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), min(vals), max(vals)


# ---------------------------------------------------------------- Table 4 style
PAPER_T4 = {
    "EEQA": (32.09, 33.77, 32.91, 25.85, 27.20, 26.51),
    "DEGREE": (67.79, 19.13, 29.84, 48.99, 13.83, 21.57),
    "OneIE": (51.11, 56.29, 53.57, 39.69, 43.71, 41.61),
    "GPT (0-shot)": (43.03, 55.56, 48.50, 30.40, 39.25, 34.26),
    "GPT (1-shot)": (50.14, 50.22, 50.18, 34.60, 34.66, 34.63),
    "GPT (2-shot)": (49.12, 51.29, 50.18, 33.99, 35.49, 34.72),
    "GPT (5-shot)": (50.04, 49.93, 49.98, 34.51, 34.42, 34.47),
    "Qwen (5-shot)": (46.94, 31.36, 37.60, 21.67, 14.48, 17.36),
    "Llama (1-shot)": (44.70, 34.08, 38.68, 18.93, 14.44, 16.38),
    "DS-R1-Llama (1-shot)": (42.62, 17.67, 24.98, 19.59, 8.12, 11.48),
}
print("=" * 104)
print("TABLE 4 COMPARISON — IoU-based argument identification / classification (%)")
print("=" * 104)
print(f"{'Method':24s} | {'ArgI-P':>7} {'ArgI-R':>7} {'ArgI-F1':>8} | {'ArgC-P':>7} {'ArgC-R':>7} {'ArgC-F1':>8}")
print("-" * 104)
for k, v in PAPER_T4.items():
    print(f"{k:24s} | {v[0]:7.2f} {v[1]:7.2f} {v[2]:8.2f} | {v[3]:7.2f} {v[4]:7.2f} {v[5]:8.2f}")
print("-" * 104)
ip = agg(lambda m: m["arg_i_iou"]["p"]); ir = agg(lambda m: m["arg_i_iou"]["r"]); i1 = agg(lambda m: m["arg_i_iou"]["f1"])
cp = agg(lambda m: m["arg_c_iou"]["p"]); cr = agg(lambda m: m["arg_c_iou"]["r"]); c1 = agg(lambda m: m["arg_c_iou"]["f1"])
print(f"{'SciEvent-Next (ours)':24s} | {ip[0]:7.2f} {ir[0]:7.2f} {i1[0]:8.2f} | {cp[0]:7.2f} {cr[0]:7.2f} {c1[0]:8.2f}")
print(f"{'  +- std over seeds':24s} | {ip[1]:7.2f} {ir[1]:7.2f} {i1[1]:8.2f} | {cp[1]:7.2f} {cr[1]:7.2f} {c1[1]:8.2f}")
print(f"{'  worst seed':24s} | {ip[2]:7.2f} {ir[2]:7.2f} {i1[2]:8.2f} | {cp[2]:7.2f} {cr[2]:7.2f} {c1[2]:8.2f}")
print("-" * 104)
print(f"{'DELTA vs OneIE (best)':24s} | {ip[0]-51.11:+7.2f} {ir[0]-56.29:+7.2f} {i1[0]-53.57:+8.2f} | "
      f"{cp[0]-39.69:+7.2f} {cr[0]-43.71:+7.2f} {c1[0]-41.61:+8.2f}")

# ---------------------------------------------------------------- Table 3 style
PAPER_T3 = {
    "EEQA": (81.93, 34.57, 45.05), "DEGREE": (64.56, 63.49, 56.85), "OneIE": (73.73, 79.40, 72.40),
    "GPT (0-shot)": (65.38, 72.73, 67.57), "GPT (1-shot)": (72.67, 77.77, 74.05),
    "GPT (2-shot)": (73.38, 78.45, 74.76), "GPT (5-shot)": (73.70, 78.82, 75.08),
    "Qwen (2-shot)": (57.27, 69.71, 61.18), "Llama (0-shot)": (54.88, 61.07, 55.83),
    "DS-R1-Llama (1-shot)": (41.81, 41.94, 40.72),
}
print("\n" + "=" * 104)
print("TABLE 3 COMPARISON — trigger identification, ROUGE-L (%)")
print("=" * 104)
print(f"{'Method':24s} | {'P':>7} {'R':>7} {'F1':>8}")
print("-" * 104)
for k, v in PAPER_T3.items():
    print(f"{k:24s} | {v[0]:7.2f} {v[1]:7.2f} {v[2]:8.2f}")
print("-" * 104)
rp = agg(lambda m: m["trigger_rougeL"]["p"]); rr = agg(lambda m: m["trigger_rougeL"]["r"]); r1 = agg(lambda m: m["trigger_rougeL"]["f1"])
print(f"{'SciEvent-Next (ours)':24s} | {rp[0]:7.2f} {rr[0]:7.2f} {r1[0]:8.2f}")
print(f"{'  +- std over seeds':24s} | {rp[1]:7.2f} {rr[1]:7.2f} {r1[1]:8.2f}")
print(f"{'DELTA vs GPT 5-shot':24s} | {rp[0]-73.70:+7.2f} {rr[0]-78.82:+7.2f} {r1[0]-75.08:+8.2f}")

# ---------------------------------------------------- everything else we track
print("\n" + "=" * 104)
print("ALL MATCHING MODES the official evaluator reports (ours, mean +- std)")
print("=" * 104)
print(f"{'mode':16s} | {'ArgI-P':>7} {'ArgI-R':>7} {'ArgI-F1':>8} | {'ArgC-P':>7} {'ArgC-R':>7} {'ArgC-F1':>8}")
print("-" * 104)
for mode, lab in [("exact", "Exact Match"), ("overlap", "Simple overlap"), ("scirex", "SciREX >0.5"), ("iou", "IoU >0.5")]:
    a = [agg(lambda m, k=f"arg_i_{mode}", f=x: m[k][f]) for x in ["p", "r", "f1"]]
    b = [agg(lambda m, k=f"arg_c_{mode}", f=x: m[k][f]) for x in ["p", "r", "f1"]]
    print(f"{lab:16s} | {a[0][0]:7.2f} {a[1][0]:7.2f} {a[2][0]:8.2f} | {b[0][0]:7.2f} {b[1][0]:7.2f} {b[2][0]:8.2f}")
    print(f"{'  +- std':16s} | {a[0][1]:7.2f} {a[1][1]:7.2f} {a[2][1]:8.2f} | {b[0][1]:7.2f} {b[1][1]:7.2f} {b[2][1]:8.2f}")

for name, key in [("PER-DOMAIN", "domain_iou"), ("PER-EVENT-TYPE", "eventtype_iou")]:
    print("\n" + "=" * 104)
    print(f"{name} Arg-I / Arg-C IoU F1 (mean +- std over seeds)")
    print("=" * 104)
    keys = sorted(mets[0][key]["Arg_C"])
    print(f"{'key':28s} | {'ArgI-F1':>16} | {'ArgC-F1':>16} | gold")
    for k in keys:
        fi = [prf(m[key]["Arg_I"][k])[2] for m in mets]
        fc = [prf(m[key]["Arg_C"][k])[2] for m in mets]
        g = mets[0][key]["Arg_C"][k]["gold_total"]
        print(f"{k:28s} | {st.mean(fi):7.2f} +-{st.stdev(fi) if len(fi)>1 else 0:5.2f} | "
              f"{st.mean(fc):7.2f} +-{st.stdev(fc) if len(fc)>1 else 0:5.2f} | {g}")

print("\n" + "=" * 104)
print("PER-ROLE Arg-C IoU P/R/F1 (mean over seeds)")
print("=" * 104)
print(f"{'role':18s} | {'P':>7} {'R':>7} {'F1':>8} | gold")
for r in sorted(mets[0]["rolewise_iou"]):
    vals = [prf(m["rolewise_iou"][r]) for m in mets]
    g = mets[0]["rolewise_iou"][r]["gold_total"]
    print(f"{r:18s} | {st.mean([v[0] for v in vals]):7.2f} {st.mean([v[1] for v in vals]):7.2f} "
          f"{st.mean([v[2] for v in vals]):8.2f} | {g}")
