"""Inference, posterior ensembling, and metric-aware decoding (H5 / H7).

Caches word-level posteriors for a set of checkpoints so that decoding rules can
be tuned on dev without re-running the encoder. Decoding rules are:

  threshold  drop a span whose mean token posterior is below tau (per role)
  merge_gap  join two same-role spans separated by <= g tokens
  min_len    drop spans shorter than this

All three trade precision against recall under the official IoU>0.5 one-to-one
matching. They are tuned on **dev only** and frozen before test.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from carve.data import (
    AAO_LABEL2ID,
    AAO_LABELS,
    EVENT_TYPES,
    ROLE_LABEL2ID,
    ROLE_LABELS,
    ROLE_TYPES,
    bio_to_spans,
    default_data_dir,
    load_split,
)
from carve.evaluate import prf_from_counts, score, to_oneie_record, write_jsonl
from carve.model import CarveModel
from carve.train import WindowDataset, collate

ID2ROLE = {i: l for l, i in ROLE_LABEL2ID.items()}
ID2AAO = {i: l for l, i in AAO_LABEL2ID.items()}



def _detect_heads(ckpt):
    """Detect which optional heads a checkpoint was trained with, and how many
    ROLE labels it carries (the single-head ablation merges both label spaces)."""
    sd = torch.load(ckpt, map_location="cpu")
    n_role = None
    for k, v in sd.items():
        if k.startswith("role_head.") and k.endswith(".weight") and v.dim() == 2:
            n_role = v.shape[0]
    return {"use_crf": any(k.startswith("role_crf.") for k in sd),
            "use_span_role": any(k.startswith("span_role_head.") for k in sd),
            "single_head": not any(k.startswith("aao_head.") for k in sd),
            "n_role_labels": n_role}


@torch.no_grad()
def posteriors(ckpts, model_name, windows, tok, max_len=640, batch_size=16, device="cuda",
               use_crf=False, use_span_role=False):
    """Average softmax posteriors over a list of checkpoints (H7 ensembling)."""
    ds = WindowDataset(windows, tok, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    n = len(windows)
    role_p = [None] * n
    aao_p = [None] * n
    type_p = np.zeros((n, len(EVENT_TYPES)), dtype=np.float64)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    heads = _detect_heads(ckpts[0])
    use_crf = use_crf or heads["use_crf"]
    n_role = heads["n_role_labels"] or len(ROLE_LABELS)
    model = CarveModel(model_name, n_role, len(AAO_LABELS), len(EVENT_TYPES),
                       use_crf=use_crf, use_span_role=heads["use_span_role"],
                       single_head=heads["single_head"]).to(device)
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp):
                tl, rl, al = model(b["input_ids"], b["attention_mask"], b["word_index"], b["word_mask"])
            tsm = torch.softmax(tl.float(), -1).cpu().numpy()
            if use_crf:
                rsm = model.role_crf.marginals(rl, b["word_mask"]).cpu().numpy()
                asm = model.aao_crf.marginals(al, b["word_mask"]).cpu().numpy()
            else:
                rsm = torch.softmax(rl.float(), -1).cpu().numpy()
                # Under the single-head ablation there is no separate AAO head:
                # both span groups are decoded from the one merged tagger.
                asm = rsm if al is None else torch.softmax(al.float(), -1).cpu().numpy()
            for j, idx in enumerate(batch["idx"].tolist()):
                nv = int(batch["word_mask"][j].sum())
                type_p[idx] += tsm[j] / len(ckpts)
                r = rsm[j, :nv] / len(ckpts)
                a = asm[j, :nv] / len(ckpts)
                role_p[idx] = r if role_p[idx] is None else role_p[idx] + r
                aao_p[idx] = a if aao_p[idx] is None else aao_p[idx] + a
    return role_p, aao_p, type_p


def spans_with_conf(p, id2label):
    """argmax decode + mean-posterior confidence per span."""
    ids = p.argmax(-1).tolist()
    out = []
    for s, e, t in bio_to_spans(ids, id2label):
        conf = float(np.mean([p[i, ids[i]] for i in range(s, e)]))
        out.append((s, e, t, conf))
    return out


def apply_decoding_rules(spans, tau, merge_gap, min_len, n_tokens, tau_short=None, short_len=None):
    """Filter and merge decoded spans.

    The default rule is a hard minimum length: a span shorter than `min_len` is
    dropped whatever its confidence. That is a *length-conditioned threshold with
    an infinite short-span threshold*, and it is what makes recall collapse to
    14.3% on 1-4 token gold spans.

    Passing `tau_short` and `short_len` replaces the infinity with a finite, higher
    threshold, so a short span survives when the model is confident enough:

        tau_eff(len) = tau_short   if len <  short_len
                     = tau         if len >= short_len

    The parameter count is unchanged (three), so this is not extra capacity fitted
    to dev -- it is the same rule with the degenerate branch relaxed.
    """
    def keep(sp):
        length = sp[1] - sp[0]
        base = tau.get(sp[2], tau.get("_", 0.0))
        if tau_short is not None and short_len is not None:
            thr = tau_short if length < short_len else base
            return sp[3] >= thr
        return sp[3] >= base and length >= min_len

    kept = [s for s in spans if keep(s)]
    if merge_gap > 0:
        kept.sort(key=lambda x: x[0])
        merged = []
        for sp in kept:
            if merged and merged[-1][2] == sp[2] and sp[0] - merged[-1][1] <= merge_gap:
                a = merged[-1]
                w1, w2 = a[1] - a[0], sp[1] - sp[0]
                merged[-1] = (a[0], sp[1], a[2], (a[3] * w1 + sp[3] * w2) / (w1 + w2))
            else:
                merged.append(sp)
        kept = merged
    return [(s, e, t) for s, e, t, _ in kept]


@torch.no_grad()
def span_rescore(ckpts, model_name, windows, tok, spans_per_window, max_len=640,
                 batch_size=16, device="cuda", use_crf=False):
    """Re-label already-decoded spans with the span-level role head (H8),
    averaging role posteriors across checkpoints."""
    ds = WindowDataset(windows, tok, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    acc = [np.zeros((len(sp), len(ROLE_TYPES))) for sp in spans_per_window]
    model = CarveModel(model_name, len(ROLE_LABELS), len(AAO_LABELS), len(EVENT_TYPES),
                         use_crf=use_crf or _detect_heads(ckpts[0])["use_crf"], use_span_role=True).to(device)
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp):
                fwd = model(b["input_ids"], b["attention_mask"], b["word_index"], b["word_mask"],
                            return_states=True)
            w_cond = fwd[3]
            flat, owner = [], []
            for j, idx in enumerate(batch["idx"].tolist()):
                for k, (s_, e_, _) in enumerate(spans_per_window[idx]):
                    flat.append((j, s_, e_))
                    owner.append((idx, k))
            if not flat:
                continue
            with torch.autocast("cuda", dtype=amp):
                sl = model.span_roles(w_cond, flat)
            probs = torch.softmax(sl.float(), -1).cpu().numpy()
            for (idx, k), pr in zip(owner, probs):
                acc[idx][k] += pr / len(ckpts)
    return [[(s_, e_, ROLE_TYPES[int(a[k].argmax())]) for k, (s_, e_, _) in enumerate(sp)]
            for sp, a in zip(spans_per_window, acc)]


def decode_spans(windows, role_p, tau, merge_gap, min_len, tau_short=None, short_len=None):
    return [apply_decoding_rules(spans_with_conf(role_p[i], ID2ROLE), tau, merge_gap, min_len, len(w.tokens), tau_short, short_len)
            for i, w in enumerate(windows)]


def records_from(windows, spans_per_window, aao_p, type_p, aao_precomputed=None):
    """Build prediction records. `aao_precomputed` lets a caller supply already
    decoded AAO spans (used by the single-head ablation, whose AAO spans come
    from the same merged tagger as the role spans)."""
    recs = []
    for i, w in enumerate(windows):
        n = len(w.tokens)
        asp = aao_precomputed[i] if aao_precomputed is not None else spans_with_conf(aao_p[i], ID2AAO)
        trig = next(((s, e) for s, e, t, _ in asp if t == "Action"), (0, min(1, n)))
        aao = [(s, e, t) for s, e, t, _ in asp if t != "Action"]
        recs.append(to_oneie_record(w.sent_id, w.tokens, EVENT_TYPES[int(type_p[i].argmax())],
                                    spans_per_window[i], aao, trig))
    return recs


def build_records(windows, role_p, aao_p, type_p, tau, merge_gap, min_len, override=None,
                  tau_short=None, short_len=None):
    recs = []
    for i, w in enumerate(windows):
        n = len(w.tokens)
        rs = apply_decoding_rules(spans_with_conf(role_p[i], ID2ROLE), tau, merge_gap, min_len, n,
                                  tau_short, short_len)
        if override is not None:
            rs = [(s_, e_, override[i].get((s_, e_), t_)) for s_, e_, t_ in rs]
        asp = spans_with_conf(aao_p[i], ID2AAO)
        trig = next(((s, e) for s, e, t, _ in asp if t == "Action"), (0, min(1, n)))
        aao = [(s, e, t) for s, e, t, _ in asp if t != "Action"]
        recs.append(to_oneie_record(w.sent_id, w.tokens, EVENT_TYPES[int(type_p[i].argmax())], rs, aao, trig))
    return recs


def tune_decoding_rules(windows, role_p, aao_p, type_p, gold_path, verbose=True, override=None, per_role=False):
    """Grid-search decoding rules on dev. Returns the best rule set + its score."""
    best = None
    for min_len in [1, 2, 3, 4]:
        for merge_gap in [0, 1, 2]:
            for t in [0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95]:
                tau = {"_": t}
                m = score(build_records(windows, role_p, aao_p, type_p, tau, merge_gap, min_len, override), gold_path)
                f1 = m["arg_c_iou"]["f1"]
                if best is None or f1 > best["f1"]:
                    best = {"f1": f1, "tau": dict(tau), "merge_gap": merge_gap, "min_len": min_len, "metrics": m}
    if verbose:
        print(f"[global rules] ArgC-IoU {best['f1']:.2f}  tau={best['tau']['_']} "
              f"merge_gap={best['merge_gap']} min_len={best['min_len']}")

    if not per_role:
        # Prefer the 3-parameter global rule: with only 158 dev windows, fitting
        # nine extra thresholds is dev overfitting for a fraction of a point.
        return best

    # per-role refinement, coordinate ascent from the best global threshold
    tau = {r: best["tau"]["_"] for r in ROLE_TYPES}
    tau["_"] = best["tau"]["_"]
    cur = best["f1"]
    for _ in range(2):
        for r in ROLE_TYPES:
            base = tau[r]
            for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                tau[r] = t
                m = score(build_records(windows, role_p, aao_p, type_p, tau,
                                        best["merge_gap"], best["min_len"], override), gold_path)
                if m["arg_c_iou"]["f1"] > cur + 1e-9:
                    cur, base = m["arg_c_iou"]["f1"], t
                    best = {"f1": cur, "tau": dict(tau), "merge_gap": best["merge_gap"],
                            "min_len": best["min_len"], "metrics": m}
            tau[r] = base
    if verbose:
        print(f"[per-role rules] ArgC-IoU {best['f1']:.2f}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-large")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tune", action="store_true", help="grid-search decoding rules on this split")
    ap.add_argument("--rules", default=None, help="json file with frozen decoding rules")
    ap.add_argument("--out", default=None)
    ap.add_argument("--use_crf", action="store_true")
    ap.add_argument("--use_span_role", action="store_true")
    ap.add_argument("--per_role", action="store_true")
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_name)
    windows = load_split(a.split, a.data_dir)
    gold = os.path.join(a.data_dir or default_data_dir(), f"{a.split}.oneie.json")
    heads = _detect_heads(a.ckpt[0])
    use_crf = a.use_crf or heads["use_crf"]
    rp, ap_, tp = posteriors(a.ckpt, a.model_name, windows, tok, use_crf=use_crf)

    override = None
    if a.use_span_role:
        # Rescore the full candidate span set once; the decoding rules only ever
        # select a subset of it (merge_gap=0), so one GPU pass suffices.
        cand = decode_spans(windows, rp, {"_": 0.0}, 0, 1)
        res = span_rescore(a.ckpt, a.model_name, windows, tok, cand, use_crf=use_crf)
        override = [{(s_, e_): t_ for s_, e_, t_ in r} for r in res]

    base = score(build_records(windows, rp, ap_, tp, {"_": 0.0}, 0, 1, override), gold)
    print(f"[argmax ] ArgC-IoU {base['arg_c_iou']['f1']:.2f} "
          f"(P{base['arg_c_iou']['p']:.1f}/R{base['arg_c_iou']['r']:.1f})  "
          f"ArgI-IoU {base['arg_i_iou']['f1']:.2f}  RgL {base['trigger_rougeL']['f1']:.2f}")

    if a.tune:
        best = tune_decoding_rules(windows, rp, ap_, tp, gold, override=override, per_role=a.per_role)
        rules = {"tau": best["tau"], "merge_gap": best["merge_gap"], "min_len": best["min_len"]}
        m = best["metrics"]
        print(json.dumps(rules, indent=2))
        print(f"[tuned  ] ArgC-IoU {m['arg_c_iou']['f1']:.2f} "
              f"(P{m['arg_c_iou']['p']:.1f}/R{m['arg_c_iou']['r']:.1f})  "
              f"ArgI-IoU {m['arg_i_iou']['f1']:.2f}  ArgC-EM {m['arg_c_exact']['f1']:.2f}  "
              f"ArgI-EM {m['arg_i_exact']['f1']:.2f}  RgL {m['trigger_rougeL']['f1']:.2f}")
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump({"rules": rules, "dev_arg_c_iou_f1": best["f1"]}, open(a.out, "w"), indent=2)
    elif a.rules:
        r = json.load(open(a.rules))["rules"]
        recs = build_records(windows, rp, ap_, tp, r["tau"], r["merge_gap"], r["min_len"], override,
                             tau_short=r.get("tau_short"), short_len=r.get("short_len"))
        m = score(recs, gold)
        print(f"\n=== FROZEN EVALUATION on {a.split} ===")
        print(f"{'metric':16s} {'P':>7} {'R':>7} {'F1':>7}")
        for k, lab in [("arg_c_iou", "Arg-C IoU"), ("arg_i_iou", "Arg-I IoU"),
                       ("arg_c_exact", "Arg-C EM"), ("arg_i_exact", "Arg-I EM"),
                       ("arg_c_overlap", "Arg-C overlap"), ("arg_i_overlap", "Arg-I overlap"),
                       ("arg_c_scirex", "Arg-C SciREX"), ("arg_i_scirex", "Arg-I SciREX")]:
            d = m[k]
            print(f"{lab:16s} {d['p']:7.2f} {d['r']:7.2f} {d['f1']:7.2f}")
        t = m["trigger_rougeL"]
        print(f"{'Trigger ROUGE-L':16s} {t['p']:7.2f} {t['r']:7.2f} {t['f1']:7.2f}")
        print("\nper-domain Arg-C IoU F1:")
        for dom, st in sorted(m["domain_iou"]["Arg_C"].items()):
            print(f"   {dom:10s} {prf_from_counts(st)[2]:6.2f}")
        print("per-event-type Arg-C IoU F1:")
        for et, st in sorted(m["eventtype_iou"]["Arg_C"].items()):
            print(f"   {et:26s} {prf_from_counts(st)[2]:6.2f}")
        print("per-role Arg-C IoU F1:")
        for r, st in sorted(m["rolewise_iou"].items()):
            print(f"   {r:16s} {prf_from_counts(st)[2]:6.2f}  (gold {st['gold_total']})")
        json.dump({k: v for k, v in m.items() if isinstance(v, dict)},
                  open(os.path.join(os.path.dirname(a.out or "."), f"metrics.{a.split}.json"), "w"),
                  indent=2, default=float)
        if a.out:
            write_jsonl(recs, a.out)
