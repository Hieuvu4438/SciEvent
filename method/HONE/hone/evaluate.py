"""Official-metric wrapper.

We do NOT reimplement the metrics. We import the upstream scorer
(third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py) and call its own
functions, so matching semantics (event-type sensitivity, trigger insensitivity,
role exclusion, IoU>0.5, the ROUGE-L <Agent, trigger, PrimaryObject,
SecondaryObject> tuple) are bit-identical to the numbers in the paper.
"""

import importlib.util
import json
import os

from hone.paths import official_scorer_path

_SPEC_CACHE = {}


def official_scorer():
    """Load the benchmark's own scorer as a module and cache it."""
    if "mod" not in _SPEC_CACHE:
        path = official_scorer_path()
        spec = importlib.util.spec_from_file_location("scievent_official_scorer", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SPEC_CACHE["mod"] = mod
    return _SPEC_CACHE["mod"]


def to_oneie_record(sent_id, tokens, event_type, role_spans, aao_spans, trigger_span):
    """Build one prediction line in the exact schema the upstream evaluator reads."""
    ents, args = [], []
    for i, (s, e, t) in enumerate(list(role_spans) + list(aao_spans)):
        eid = f"{sent_id}-P{i}"
        ents.append({"id": eid, "start": int(s), "end": int(e), "entity_type": "UNK", "mention_type": "UNK"})
        args.append({"entity_id": eid, "role": t})
    ts, te = trigger_span
    return {
        "doc_id": sent_id.rsplit("-", 1)[0],
        "sent_id": sent_id,
        "tokens": tokens,
        "entity_mentions": ents,
        "relation_mentions": [],
        "event_mentions": [
            {
                "event_type": event_type,
                "id": f"{sent_id}-EV0",
                "trigger": {"start": int(ts), "end": int(te)},
                "arguments": args,
            }
        ],
        "entity_coreference": [],
        "event_coreference": [],
    }


def score(pred_records, gold_path, with_rouge=True):
    """Return the full official metric dict for a list of prediction records.

    `with_rouge=False` skips the ROUGE-L block. That block is independent of the
    decoding threshold (the AAO head is never thresholded), so recomputing it at
    every point of a tau/min_len grid is pure waste. The default is unchanged, so
    every reported number comes from the full computation.
    """
    M = official_scorer()
    gold_data = M.load_jsonl(gold_path)
    tokens = {e["sent_id"]: e["tokens"] for e in gold_data}
    wnd_ids = {e["sent_id"]: e["sent_id"] for e in gold_data}

    _, pred_roles = M.extract_triggers_and_roles(pred_records)
    _, gold_roles = M.extract_triggers_and_roles(gold_data)

    sent_ids = list(gold_roles.keys())
    g_all = [gold_roles[s] for s in sent_ids]
    p_all = [pred_roles.get(s, []) for s in sent_ids]
    t_all = [tokens[s] for s in sent_ids]

    res = {}
    if with_rouge:
        rp, rr, rf = M.compute_rougeL_overall(g_all, p_all, t_all, sent_ids)
        res["trigger_rougeL"] = {"p": rp * 100, "r": rr * 100, "f1": rf * 100}

    matchers = {
        "exact": None,
        "overlap": M.spans_overlap,
        "scirex": M.scirex_overlap,
        "iou": M.iou_overlap,
    }
    for name, fn in matchers.items():
        if fn is None:
            argi = M.compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][:2] == y[1][:2])
            argc = M.compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1] == y[1])
        else:
            argi = M.compute_f1(
                pred_roles, gold_roles, lambda x, y, f=fn: x[0][2] == y[0][2] and f(x[1][:2], y[1][:2])
            )
            argc = M.compute_f1(
                pred_roles,
                gold_roles,
                lambda x, y, f=fn: x[0][2] == y[0][2] and x[1][2] == y[1][2] and f(x[1][:2], y[1][:2]),
            )
        res[f"arg_i_{name}"] = {"p": argi[0] * 100, "r": argi[1] * 100, "f1": argi[2] * 100,
                                "matched": argi[3], "pred": argi[4], "gold": argi[5]}
        res[f"arg_c_{name}"] = {"p": argc[0] * 100, "r": argc[1] * 100, "f1": argc[2] * 100,
                                "matched": argc[3], "pred": argc[4], "gold": argc[5]}

    if not with_rouge:
        return res

    res["rolewise_iou"] = {
        r: dict(v) for r, v in M.compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=M.iou_overlap).items()
    }
    et = M.compute_eventtype_f1_extended(pred_roles, gold_roles, overlap_fn=M.iou_overlap)
    res["eventtype_iou"] = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in et.items()}
    dm = M.compute_domain_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=M.iou_overlap)
    res["domain_iou"] = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in dm.items()}
    res["rougeL_domain"] = M.compute_rougeL_domain(g_all, p_all, t_all, [wnd_ids[s] for s in sent_ids])
    res["rougeL_eventtype"] = M.compute_rougeL_eventtype(g_all, p_all, t_all, sent_ids)
    return res


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def prf_from_counts(d):
    m, p, g = d["matched"], d["pred_total"], d["gold_total"]
    pr = m / p if p else 0.0
    rc = m / g if g else 0.0
    return pr * 100, rc * 100, (2 * pr * rc / (pr + rc) * 100 if pr + rc else 0.0)
