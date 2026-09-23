"""Train the HONE verifier on out-of-fold train candidates; select on dev.

Train candidates: OOF proposals for every train window, pooled over the chosen
proposer seeds. Dev candidates: proposals from the full-train proposers, pooled
over the same number of seeds, so both sides see the same candidate
distribution. Each epoch the dev candidates are decoded under a dev-tuned rule
(select.tune) and scored with the official scorer; the best epoch is kept.

Usage:
  python3 scripts/train_verifier.py --name v1 --prop_seeds 42 --seed 42
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from hone import candidates as C
from hone.data import load_split
from hone.paths import repo_root, split_path
from hone.select import records, tune
from hone.verifier import MARK_CLOSE, MARK_OPEN, Verifier, VerifierDataset, build_examples, collate

ROOT = repo_root()
K_FOLDS = 5


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_train_rows(prop_seeds, prefix=""):
    rows_by_seed = []
    for s in prop_seeds:
        rows = []
        for k in range(K_FOLDS):
            rows += C.load(str(ROOT / "data" / "cands" / f"{prefix}oof_k{k}_s{s}.jsonl"))
        rows_by_seed.append(rows)
    return C.pool_rows(rows_by_seed)


def load_eval_rows(split, prop_seeds, prefix=""):
    return C.pool_rows([C.load(str(ROOT / "data" / "cands" / f"{prefix}{split}_s{s}.jsonl")) for s in prop_seeds])


@torch.no_grad()
def predict(model, dl, n, device, amp):
    model.eval()
    out = np.zeros((n, len(C.LABELS)), dtype=np.float32)
    for b in dl:
        with torch.autocast("cuda", dtype=amp):
            lg = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                       b["open"].to(device), b["close"].to(device), b["feats"].to(device))
        out[b["idx"].numpy()] = torch.softmax(lg.float(), -1).cpu().numpy()
    return out


def regroup(examples, probs, rows):
    """Flat per-candidate probabilities -> per-window arrays aligned with rows."""
    pos = {}
    for i, x in enumerate(examples):
        pos[(x["sent_id"], x["cand_idx"])] = i
    per_w = []
    for r in rows:
        idx = [pos[(r["sent_id"], j)] for j in range(len(r["cands"]))]
        per_w.append(probs[idx] if idx else np.zeros((0, len(C.LABELS)), dtype=np.float32))
    return per_w


def main(a):
    set_seed(a.seed)
    device = "cuda"
    amp = torch.bfloat16
    run_dir = ROOT / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})

    train_w = {w.sent_id: w for w in load_split("train")}
    dev_list = load_split("dev")
    dev_w = {w.sent_id: w for w in dev_list}

    if a.smoke:   # code-path test only: train on dev candidates, numbers meaningless
        tr_rows = load_eval_rows("dev", a.prop_seeds, a.cand_prefix)[:60]
        train_w = dev_w
    elif a.insample:   # H4 control: in-sample candidates, never used by the method
        tr_rows = C.pool_rows([C.load(str(ROOT / "data" / "cands" / f"insample_s{s}.jsonl"))
                               for s in a.prop_seeds])
    else:
        tr_rows = load_train_rows(a.prop_seeds, a.cand_prefix)
    if a.compose:
        for r in tr_rows:
            r["cands"] = C.compose(r["cands"], a.compose_run, a.compose_gap)
    for r in tr_rows:
        C.label(train_w[r["sent_id"]], r["cands"])
    dv_rows = load_eval_rows("dev", a.prop_seeds, a.cand_prefix)
    if a.compose:
        for r in dv_rows:
            r["cands"] = C.compose(r["cands"], a.compose_run, a.compose_gap)
    for r in dv_rows:
        C.label(dev_w[r["sent_id"]], r["cands"])
    dv_windows = [dev_w[r["sent_id"]] for r in dv_rows]

    tr_ex = build_examples(train_w, tr_rows)
    dv_ex = build_examples(dev_w, dv_rows)
    pos = sum(x["label"] != 0 for x in tr_ex) / len(tr_ex)
    print(f"train candidates {len(tr_ex)} (positive {pos:.1%}) | dev candidates {len(dv_ex)}", flush=True)

    tr_dl = DataLoader(VerifierDataset(tr_ex, tok), batch_size=a.batch_size, shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=2)
    dv_dl = DataLoader(VerifierDataset(dv_ex, tok), batch_size=a.eval_batch_size, shuffle=False,
                       collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=2)

    model = Verifier(a.model_name, len(tok), dropout=a.dropout, use_feats=not a.no_feats).to(device)
    if a.grad_ckpt:   # trades compute for activation memory; results are unaffected
        model.encoder.gradient_checkpointing_enable()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_encoder}, {"params": head, "lr": a.lr_head}],
                            weight_decay=0.01)
    steps = (len(tr_dl) + a.accum - 1) // a.accum * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    ce = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    gold = split_path("dev")
    best, log, t0 = None, [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for step, b in enumerate(tr_dl):
            with torch.autocast("cuda", dtype=amp):
                lg = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                           b["open"].to(device), b["close"].to(device), b["feats"].to(device))
                loss = ce(lg.float(), b["label"].to(device))
            (loss / a.accum).backward()
            if (step + 1) % a.accum == 0 or step + 1 == len(tr_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item()

        probs = predict(model, dv_dl, len(dv_ex), device, amp)
        per_w = regroup(dv_ex, probs, dv_rows)
        cand_acc = float(np.mean(probs.argmax(-1) == np.array([x["label"] for x in dv_ex])))
        rule = tune(dv_windows, dv_rows, per_w, gold)
        full = __import__("hone.evaluate", fromlist=["score"]).score(
            records(dv_windows, dv_rows, per_w, rule["theta"], rule["min_len"], rule["nms"]), gold)
        row = {"epoch": ep, "loss": tot / len(tr_dl), "cand_acc": cand_acc * 100,
               "arg_c_iou_f1": full["arg_c_iou"]["f1"], "arg_c_p": full["arg_c_iou"]["p"],
               "arg_c_r": full["arg_c_iou"]["r"], "arg_i_iou_f1": full["arg_i_iou"]["f1"],
               "rougeL": full["trigger_rougeL"]["f1"], "rule": rule, "secs": round(time.time() - t0)}
        log.append(row)
        print(f"ep{ep:02d} loss {row['loss']:.3f} | cand-acc {row['cand_acc']:.1f} | "
              f"ArgC-IoU {row['arg_c_iou_f1']:.2f} (P{row['arg_c_p']:.1f}/R{row['arg_c_r']:.1f}) | "
              f"ArgI {row['arg_i_iou_f1']:.2f} | rule th={rule['theta']} ml={rule['min_len']} "
              f"nms={rule['nms']} | {row['secs']}s", flush=True)
        if best is None or row["arg_c_iou_f1"] > best["arg_c_iou_f1"]:
            best = dict(row)
            np.save(run_dir / "dev_probs.npy", probs)
            if a.save_model:
                torch.save(model.state_dict(), run_dir / "best.pt")

    json.dump({"args": vars(a), "log": log, "best": best}, open(run_dir / "log.json", "w"), indent=2)
    print("BEST", json.dumps({k: v for k, v in best.items()}, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--prop_seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-large")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr_encoder", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--save_model", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--insample", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--compose", action="store_true")
    ap.add_argument("--cand_prefix", default="", help="'simple_' = CARVE-simple proposer")
    ap.add_argument("--compose_run", type=int, default=3)
    ap.add_argument("--compose_gap", type=int, default=2)
    main(ap.parse_args())
