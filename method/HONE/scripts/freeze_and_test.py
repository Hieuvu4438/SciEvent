"""Freeze HONE's decoding rule on dev, then (separately) evaluate test once.

  freeze  Grid over (theta, min_len, nms, type_lambda). The criterion is the MEAN
          dev Arg-C IoU across verifier seeds, so the rule is not fitted to
          whichever seed happens to look best. Writes assets/decoding_rule.json.

  test    Loads the frozen rule, runs each verifier seed on the pooled test
          candidates, decodes with that one rule, scores with the official
          scorer, writes predictions. Run only after `freeze`. The rule file's
          timestamp precedes every test prediction file by construction.

Usage:
  python3 scripts/freeze_and_test.py freeze --runs v_s42 v_s13 v_s101 --prop_seeds 42 13 101
  python3 scripts/freeze_and_test.py test   --runs v_s42 v_s13 v_s101 --prop_seeds 42 13 101
"""

import argparse
import json
import statistics as st
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hone import candidates as C
from hone.data import load_split
from hone.evaluate import score, write_jsonl
from hone.paths import repo_root, split_path
from hone.select import ALPHAS, MIN_LENS, NMS_MODES, THETAS, mix_roles, records, role_given_type
from hone.verifier import MARK_CLOSE, MARK_OPEN, Verifier, VerifierDataset, build_examples, collate

ROOT = repo_root()
MODEL = "microsoft/deberta-v3-large"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def run_args(runs):
    """Candidate construction must be identical across the runs being averaged."""
    args = [json.load(open(ROOT / "runs" / r / "log.json"))["args"] for r in runs]
    key = lambda x: (x.get("compose", False), x.get("compose_run", 3), x.get("compose_gap", 2),
                     x.get("cand_prefix", ""))
    assert len({key(x) for x in args}) == 1, "runs differ in candidate composition"
    return args[0]


def pooled_rows(split, prop_seeds, windows_by_id, args=None):
    pre = (args or {}).get("cand_prefix", "")
    rows = C.pool_rows([C.load(str(ROOT / "data" / "cands" / f"{pre}{split}_s{s}.jsonl")) for s in prop_seeds])
    if args and args.get("compose"):
        for r in rows:
            r["cands"] = C.compose(r["cands"], args["compose_run"], args["compose_gap"])
    return rows, [windows_by_id[r["sent_id"]] for r in rows]


def regroup(examples, probs, rows):
    pos = {(x["sent_id"], x["cand_idx"]): i for i, x in enumerate(examples)}
    return [probs[[pos[(r["sent_id"], j)] for j in range(len(r["cands"]))]] if r["cands"]
            else np.zeros((0, len(C.LABELS)), dtype=np.float32) for r in rows]


def cmd_freeze(a):
    dev_by = {w.sent_id: w for w in load_split("dev")}
    rows, wins = pooled_rows("dev", a.prop_seeds, dev_by, run_args(a.runs))
    ex = build_examples(dev_by, rows, with_labels=False)
    flat = [np.load(ROOT / "runs" / r / "dev_probs.npy") for r in a.runs]
    if a.ensemble:   # H5: one predictor = mean verifier probabilities over runs
        flat = [np.mean(flat, axis=0)]
    per_run = [regroup(ex, f, rows) for f in flat]
    lrt = role_given_type(load_split("train"))
    gold = split_path("dev")
    best = None
    for alpha in ALPHAS:
        mixed = [mix_roles(rows, pw, alpha) for pw in per_run]
        for lam in LAMBDAS:
            for nms in NMS_MODES:
                for ml in MIN_LENS:
                    for th in THETAS:
                        fs = [score(records(wins, rows, pw, th, ml, nms, lam, lrt), gold,
                                    with_rouge=False)["arg_c_iou"]["f1"] for pw in mixed]
                        mu = st.mean(fs)
                        if best is None or mu > best["mean"]:
                            best = {"mean": mu, "std": st.stdev(fs) if len(fs) > 1 else 0.0,
                                    "per_run": fs, "theta": th, "min_len": ml, "nms": nms,
                                    "type_lambda": lam, "role_alpha": alpha}
        print(f"alpha={alpha}: best so far {best['mean']:.2f} (alpha {best['role_alpha']})", flush=True)
    rule = {"rule": {k: best[k] for k in ["theta", "min_len", "nms", "type_lambda", "role_alpha"]},
            "selection": "max MEAN dev Arg-C IoU over verifier runs",
            "runs": a.runs, "prop_seeds": a.prop_seeds, "ensemble": a.ensemble,
            "dev_arg_c_iou_mean": best["mean"], "dev_arg_c_iou_std": best["std"],
            "dev_per_run": best["per_run"], "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = ROOT / "assets" / a.rule_name
    out.parent.mkdir(exist_ok=True)
    json.dump(rule, open(out, "w"), indent=2)
    print(json.dumps(rule, indent=2))


@torch.no_grad()
def cmd_test(a):
    rule = json.load(open(ROOT / "assets" / a.rule_name))
    R = rule["rule"]
    print(f"frozen rule (frozen at {rule['frozen_at']}): {R}")
    test_by = {w.sent_id: w for w in load_split("test")}
    rows, wins = pooled_rows("test", a.prop_seeds, test_by, run_args(a.runs))
    ex = build_examples(test_by, rows, with_labels=False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})
    dl = DataLoader(VerifierDataset(ex, tok), batch_size=32, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    lrt = role_given_type(load_split("train"))
    gold = split_path("test")
    results, all_probs = [], []
    for run in a.runs:
        args = json.load(open(ROOT / "runs" / run / "log.json"))["args"]
        sd = torch.load(ROOT / "runs" / run / "best.pt", map_location="cuda")
        nf = sd["feat.0.weight"].shape[1] if "feat.0.weight" in sd else None
        model = Verifier(MODEL, len(tok), use_feats=not args.get("no_feats", False),
                         **({"n_feats": nf} if nf else {})).cuda()
        model.load_state_dict(sd)
        model.eval()
        probs = np.zeros((len(ex), len(C.LABELS)), dtype=np.float32)
        for b in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(b["input_ids"].cuda(), b["attention_mask"].cuda(),
                           b["open"].cuda(), b["close"].cuda(), b["feats"].cuda())
            probs[b["idx"].numpy()] = torch.softmax(lg.float(), -1).cpu().numpy()
        np.save(ROOT / "runs" / run / "test_probs.npy", probs)
        all_probs.append(probs)
        del model
        torch.cuda.empty_cache()
    if rule.get("ensemble"):
        all_probs, names = [np.mean(all_probs, axis=0)], ["ensemble_" + "_".join(a.runs)]
    else:
        names = a.runs
    (ROOT / "preds").mkdir(exist_ok=True)
    for name, probs in zip(names, all_probs):
        pw = mix_roles(rows, regroup(ex, probs, rows), R.get("role_alpha", 1.0))
        recs = records(wins, rows, pw, R["theta"], R["min_len"], R["nms"],
                       R["type_lambda"], lrt)
        write_jsonl(recs, str(ROOT / "preds" / f"test_{name}.jsonl"))
        m = score(recs, gold)
        results.append(m)
        print(f"{name}: Arg-C IoU {m['arg_c_iou']['f1']:.2f} (P{m['arg_c_iou']['p']:.1f}/R{m['arg_c_iou']['r']:.1f})"
              f"  Arg-I IoU {m['arg_i_iou']['f1']:.2f}  ROUGE-L {m['trigger_rougeL']['f1']:.2f}")
    if len(results) > 1:
        for k, lab in [("arg_c_iou", "Arg-C IoU"), ("arg_i_iou", "Arg-I IoU"), ("trigger_rougeL", "ROUGE-L")]:
            v = [m[k]["f1"] for m in results]
            print(f"{lab:10s} {st.mean(v):.2f} ± {st.stdev(v):.2f}   per run {[round(x, 2) for x in v]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "test"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--prop_seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rule_name", default="decoding_rule.json")
    ap.add_argument("--ensemble", action="store_true")
    a = ap.parse_args()
    cmd_freeze(a) if a.cmd == "freeze" else cmd_test(a)
