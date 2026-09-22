"""Error analysis for a SciEvent-Next prediction file.

Produces the breakdown the research protocol requires: per-domain, per-event-type,
per-role, per-span-length, plus an error taxonomy that separates the four failure
modes we care about (they need different fixes):

  wrong_type      - the window's event type was misclassified, so every argument
                    in it is dead regardless of span quality
  missed          - a gold span with no overlapping prediction at all
  boundary        - a prediction overlaps the right gold span with the right role
                    but IoU <= 0.5  (a boundary problem)
  role_confusion  - span matches by IoU but the role label is wrong
  spurious        - a prediction overlapping no gold span
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scievent_next.data import EXCLUDED_FROM_SCORING, default_data_dir  # noqa: E402
from scievent_next.evaluate import upstream  # noqa: E402


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / (max(a[1], b[1]) - min(a[0], b[0]))


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def args_of(rec):
    ent = {e["id"]: (e["start"], e["end"]) for e in rec["entity_mentions"]}
    ev = rec["event_mentions"][0]
    out = []
    for a in ev["arguments"]:
        if a["role"] in EXCLUDED_FROM_SCORING:
            continue
        s = ent.get(a["entity_id"])
        if s:
            out.append((s[0], s[1], a["role"]))
    return ev["event_type"], out


def bucket(n):
    for hi, name in [(4, "1-4"), (9, "5-9"), (19, "10-19"), (34, "20-34")]:
        if n <= hi:
            return name
    return "35+"


def diagnose(pred_path, gold_path):
    P = {r["sent_id"]: r for r in load(pred_path)}
    G = load(gold_path)

    tax = collections.Counter()
    conf = collections.Counter()
    by_len = collections.defaultdict(lambda: {"gold": 0, "hit": 0})
    len_err = []
    type_conf = collections.Counter()
    wrong_type_gold = 0
    total_gold = 0

    for g in G:
        sid = g["sent_id"]
        getype, gargs = args_of(g)
        total_gold += len(gargs)
        if sid not in P:
            tax["missed"] += len(gargs)
            wrong_type_gold += 0
            continue
        petype, pargs = args_of(P[sid])
        type_conf[(getype, petype)] += 1
        type_ok = petype == getype

        used = set()
        for gs, ge, gr in gargs:
            b = bucket(ge - gs)
            by_len[b]["gold"] += 1
            best, bi = 0.0, -1
            for i, (ps, pe, pr) in enumerate(pargs):
                if i in used:
                    continue
                v = iou((ps, pe), (gs, ge))
                if v > best:
                    best, bi = v, i
            if not type_ok:
                tax["wrong_type"] += 1
                wrong_type_gold += 1
                continue
            if bi < 0 or best == 0.0:
                tax["missed"] += 1
                continue
            pr = pargs[bi][2]
            if best > 0.5 and pr == gr:
                tax["correct"] += 1
                by_len[b]["hit"] += 1
                used.add(bi)
                len_err.append((ge - gs, (pargs[bi][1] - pargs[bi][0]) - (ge - gs)))
            elif best > 0.5:
                tax["role_confusion"] += 1
                conf[(gr, pr)] += 1
                used.add(bi)
            elif pr == gr:
                tax["boundary"] += 1
                used.add(bi)
            else:
                tax["missed"] += 1

        for i, (ps, pe, pr) in enumerate(pargs):
            if i not in used and type_ok:
                if not any(iou((ps, pe), (gs, ge)) > 0.5 for gs, ge, _ in gargs):
                    tax["spurious"] += 1

    print(f"\n{'='*72}\nERROR TAXONOMY over {total_gold} gold scored arguments\n{'='*72}")
    order = ["correct", "wrong_type", "missed", "boundary", "role_confusion", "spurious"]
    for k in order:
        v = tax[k]
        base = f" ({v/total_gold*100:5.1f}% of gold)" if k != "spurious" else ""
        print(f"  {k:16s} {v:5d}{base}")
    print(f"\n  windows whose event type was wrong cost {wrong_type_gold} gold args "
          f"({wrong_type_gold/total_gold*100:.1f}% of all gold arguments)")

    print(f"\n{'='*72}\nRECALL BY GOLD SPAN LENGTH (tokens)\n{'='*72}")
    for b in ["1-4", "5-9", "10-19", "20-34", "35+"]:
        d = by_len[b]
        if d["gold"]:
            print(f"  {b:6s} n={d['gold']:4d}  recall@IoU>0.5 = {d['hit']/d['gold']*100:5.1f}")

    if len_err:
        import statistics
        d = [e for _, e in len_err]
        print(f"\n  length bias on matched spans: mean {statistics.mean(d):+.2f} tokens, "
              f"median {statistics.median(d):+.1f}")

    print(f"\n{'='*72}\nTOP ROLE CONFUSIONS (gold -> predicted)\n{'='*72}")
    for (g, p), c in conf.most_common(12):
        print(f"  {g:15s} -> {p:15s} {c}")

    print(f"\n{'='*72}\nEVENT-TYPE CONFUSION (gold -> predicted)\n{'='*72}")
    tot = sum(type_conf.values())
    acc = sum(c for (a, b), c in type_conf.items() if a == b)
    print(f"  window type accuracy: {acc/tot*100:.2f}  ({acc}/{tot})")
    for (a, b), c in sorted(type_conf.items(), key=lambda x: -x[1]):
        if a != b:
            print(f"  {a:26s} -> {b:26s} {c}")

    # official per-role / per-domain / per-type IoU tables
    M = upstream()
    preds = [P[g["sent_id"]] for g in G if g["sent_id"] in P]
    _, pr = M.extract_triggers_and_roles(preds)
    _, gr = M.extract_triggers_and_roles(G)
    wnd = {g["sent_id"]: g["sent_id"] for g in G}
    M.print_rolewise_f1_dual(M.compute_rolewise_f1_dual(pr, gr, overlap_fn=M.iou_overlap), name="IoU")
    M.print_eventtype_stats_extended("IoU", M.compute_eventtype_f1_extended(pr, gr, overlap_fn=M.iou_overlap))
    M.print_domain_f1_dual(M.compute_domain_f1_dual(pr, gr, wnd, overlap_fn=M.iou_overlap), name="IoU")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--gold", default=None)
    a = ap.parse_args()
    gold = a.gold or os.path.join(default_data_dir(), f"{a.split}.oneie.json")
    diagnose(a.pred, gold)
