"""Training / evaluation driver for SciEvent-Next."""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scievent_next.data import (  # noqa: E402
    AAO_LABEL2ID,
    AAO_LABELS,
    EVENT_TYPE2ID,
    EVENT_TYPES,
    ROLE_LABEL2ID,
    ROLE_LABELS,
    ROLE_TYPES,
    bio_to_spans,
    load_split,
    spans_to_bio,
)
from scievent_next.evaluate import score, to_oneie_record, write_jsonl  # noqa: E402
from scievent_next.model import SciEventNext  # noqa: E402

ID2ROLE = {i: l for l, i in ROLE_LABEL2ID.items()}
ID2AAO = {i: l for l, i in AAO_LABEL2ID.items()}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class WindowDataset(Dataset):
    def __init__(self, windows, tokenizer, max_len=640):
        self.w = windows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        w = self.w[i]
        enc = self.tok(
            w.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )
        word_ids = enc.word_ids()
        first_sub = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in first_sub:
                first_sub[wid] = pos
        n_words = len(w.tokens)
        word_index = [first_sub.get(j, 0) for j in range(n_words)]
        word_valid = [1 if j in first_sub else 0 for j in range(n_words)]
        role_y = spans_to_bio(w.role_spans, n_words, ROLE_LABEL2ID)
        aao_y = spans_to_bio(w.aao_spans, n_words, AAO_LABEL2ID)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "word_index": word_index,
            "word_valid": word_valid,
            "role_y": role_y,
            "aao_y": aao_y,
            "type_y": EVENT_TYPE2ID[w.event_type],
            "idx": i,
        }


def collate(batch, pad_id):
    B = len(batch)
    L = max(len(b["input_ids"]) for b in batch)
    W = max(len(b["word_index"]) for b in batch)
    out = {
        "input_ids": torch.full((B, L), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((B, L), dtype=torch.long),
        "word_index": torch.zeros((B, W), dtype=torch.long),
        "word_mask": torch.zeros((B, W), dtype=torch.float),
        "role_y": torch.full((B, W), -100, dtype=torch.long),
        "aao_y": torch.full((B, W), -100, dtype=torch.long),
        "type_y": torch.zeros(B, dtype=torch.long),
        "idx": torch.zeros(B, dtype=torch.long),
    }
    for i, b in enumerate(batch):
        l, w = len(b["input_ids"]), len(b["word_index"])
        out["input_ids"][i, :l] = torch.tensor(b["input_ids"])
        out["attention_mask"][i, :l] = torch.tensor(b["attention_mask"])
        out["word_index"][i, :w] = torch.tensor(b["word_index"])
        out["word_mask"][i, :w] = torch.tensor(b["word_valid"], dtype=torch.float)
        ry = torch.tensor(b["role_y"])
        ay = torch.tensor(b["aao_y"])
        wv = torch.tensor(b["word_valid"], dtype=torch.bool)
        ry[~wv] = -100
        ay[~wv] = -100
        out["role_y"][i, :w] = ry
        out["aao_y"][i, :w] = ay
        out["type_y"][i] = b["type_y"]
        out["idx"][i] = b["idx"]
    return out



def param_groups(model, cfg):
    """Layer-wise learning-rate decay over the encoder (standard for DeBERTa-large
    fine-tuning on small datasets: lower layers move less, which reduces the
    catastrophic-forgetting/overfitting seen with 1278 training windows)."""
    decay = cfg.get("llrd", 1.0)
    base, head_lr = cfg["lr_encoder"], cfg["lr_head"]
    try:
        n_layers = len(model.encoder.encoder.layer)
    except AttributeError:
        n_layers = len(getattr(model.encoder, "layers", []))
    groups = []
    head, enc_by_depth = [], {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not n.startswith("encoder."):
            head.append(p)
            continue
        depth = 0
        for i in range(n_layers):
            if f".layer.{i}." in n or f".layers.{i}." in n:
                depth = i + 1
                break
        enc_by_depth.setdefault(depth, []).append(p)
    for depth, ps in sorted(enc_by_depth.items()):
        groups.append({"params": ps, "lr": base * (decay ** (n_layers - depth))})
    groups.append({"params": head, "lr": head_lr})
    return groups


@torch.no_grad()
def predict(model, loader, windows, device, amp_dtype, tau=0.0, min_len=1, merge_gap=0):
    """Decode dev predictions. Returns (argmax_records, calibrated_records, type_acc).

    Checkpoint selection uses the *calibrated* records, because that is the
    decoding the final system uses; selecting on argmax F1 optimises a rule we do
    not ship.
    """
    from scievent_next.infer import apply_rules

    model.eval()
    recs_argmax, cands = [], []
    type_correct = 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        use_sr = getattr(model, "use_span_role", False)
        with torch.autocast("cuda", dtype=amp_dtype):
            fwd = model(b["input_ids"], b["attention_mask"], b["word_index"], b["word_mask"],
                        return_states=use_sr)
        tl, rl, al = fwd[0], fwd[1], fwd[2]
        tp = tl.float().argmax(-1).cpu()
        if getattr(model, "use_crf", False):
            rprob = model.role_crf.marginals(rl, b["word_mask"]).cpu().numpy()
            aprob = model.aao_crf.marginals(al, b["word_mask"]).cpu().numpy()
            rpath = model.role_crf.decode(rl, b["word_mask"])
            apath = model.aao_crf.decode(al, b["word_mask"])
        else:
            rprob = torch.softmax(rl.float(), -1).cpu().numpy()
            aprob = torch.softmax(al.float(), -1).cpu().numpy()
            rpath = apath = None

        for j, idx in enumerate(batch["idx"].tolist()):
            w = windows[idx]
            n = len(w.tokens)
            nv = int(batch["word_mask"][j, :n].sum())
            r_ids = rpath[j][:nv] if rpath is not None else rprob[j, :nv].argmax(-1).tolist()
            a_ids = apath[j][:nv] if apath is not None else aprob[j, :nv].argmax(-1).tolist()

            role_spans = bio_to_spans(r_ids, ID2ROLE)
            if use_sr and role_spans:
                sl = model.span_roles(fwd[3], [(j, s_, e_) for s_, e_, _ in role_spans])
                new_roles = sl.float().argmax(-1).cpu().tolist()
                role_spans = [(s_, e_, ROLE_TYPES[k]) for (s_, e_, _), k in zip(role_spans, new_roles)]
            aao_spans = bio_to_spans(a_ids, ID2AAO)
            etype = EVENT_TYPES[int(tp[j])]
            type_correct += int(etype == w.event_type)
            trig = next(((s, e) for s, e, t in aao_spans if t == "Action"), (0, min(1, n)))
            aao_out = [(s, e, t) for s, e, t in aao_spans if t != "Action"]
            recs_argmax.append(to_oneie_record(w.sent_id, w.tokens, etype, role_spans, aao_out, trig))

            conf = [(s, e, t, float(np.mean([rprob[j, i, r_ids[i]] for i in range(s, e)])))
                    for s, e, t in role_spans]
            cands.append({"conf": conf, "etype": etype, "aao": aao_out, "trig": trig,
                          "sent_id": w.sent_id, "tokens": w.tokens, "n": n})
    return recs_argmax, cands, type_correct / max(1, len(windows))


def records_at(cands, tau, min_len, merge_gap=0):
    from scievent_next.infer import apply_rules
    return [to_oneie_record(c["sent_id"], c["tokens"], c["etype"],
                            apply_rules(c["conf"], {"_": tau}, merge_gap, min_len, c["n"]),
                            c["aao"], c["trig"])
            for c in cands]


def run(cfg):
    set_seed(cfg["seed"])
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_w = load_split("train", cfg.get("data_dir"))
    dev_w = load_split(cfg.get("eval_split", "dev"), cfg.get("data_dir"))
    gold_path = os.path.join(
        cfg.get("data_dir") or __import__("scievent_next.data", fromlist=["x"]).default_data_dir(),
        f"{cfg.get('eval_split', 'dev')}.oneie.json",
    )

    tr_ds = WindowDataset(train_w, tok, cfg["max_len"])
    dv_ds = WindowDataset(dev_w, tok, cfg["max_len"])
    coll = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=coll, num_workers=2, drop_last=False)
    dv_dl = DataLoader(dv_ds, batch_size=cfg["eval_batch_size"], shuffle=False, collate_fn=coll, num_workers=2)

    model = SciEventNext(
        cfg["model_name"], len(ROLE_LABELS), len(AAO_LABELS), len(EVENT_TYPES),
        dropout=cfg["dropout"], use_lstm=cfg.get("use_lstm", False),
        use_crf=cfg.get("use_crf", False),
        use_span_role=cfg.get("use_span_role", False),
    ).to(device)

    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg["weight_decay"])
    steps = len(tr_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=cfg.get("label_smoothing", 0.0))
    ce_type = nn.CrossEntropyLoss()

    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    best = {"arg_c_iou_f1": -1.0}
    log = []
    t0 = time.time()

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot = 0.0
        # scheduled sampling: anneal from fully-gold conditioning to self-conditioning
        tp_p = max(cfg.get("type_teacher_min", 0.5), 1.0 - (ep - 1) / max(1, cfg["epochs"] - 1))
        for batch in tr_dl:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            use_sr = cfg.get("use_span_role", False)
            with torch.autocast("cuda", dtype=amp_dtype):
                fwd = model(
                    b["input_ids"], b["attention_mask"], b["word_index"], b["word_mask"],
                    gold_type=b["type_y"], type_teacher_p=tp_p, return_states=use_sr,
                )
                tl, rl, al = fwd[0], fwd[1], fwd[2]
                if cfg.get("use_crf", False):
                    l_role = model.role_crf(rl, b["role_y"].clamp(min=0), b["word_mask"])
                    l_aao = model.aao_crf(al, b["aao_y"].clamp(min=0), b["word_mask"])
                else:
                    l_role = ce(rl.reshape(-1, rl.size(-1)).float(), b["role_y"].reshape(-1))
                    l_aao = ce(al.reshape(-1, al.size(-1)).float(), b["aao_y"].reshape(-1))
                l_span = rl.sum() * 0.0
                if use_sr:
                    spans, tgts = [], []
                    for j in range(b["role_y"].size(0)):
                        nv = int(b["word_mask"][j].sum())
                        ids = b["role_y"][j, :nv].clamp(min=0).tolist()
                        for s_, e_, t_ in bio_to_spans(ids, ID2ROLE):
                            spans.append((j, s_, e_))
                            tgts.append(ROLE_TYPES.index(t_))
                    if spans:
                        sl = model.span_roles(fwd[3], spans)
                        l_span = ce_type(sl.float(), torch.tensor(tgts, device=sl.device))
                loss = (
                    cfg.get("w_span_role", 1.0) * l_span
                    + cfg["w_role"] * l_role
                    + cfg["w_aao"] * l_aao
                    + cfg["w_type"] * ce_type(tl.float(), b["type_y"])
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()

        recs_a, cands, type_acc = predict(model, dv_dl, dev_w, device, amp_dtype)
        m_argmax = score(recs_a, gold_path)
        # Select the checkpoint under the *same* calibrated decoding the final
        # system uses, sweeping tau so selection is not tied to a stale threshold.
        m, recs, sel_tau = None, None, None
        for t in cfg.get("dev_tau_grid", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
            r = records_at(cands, t, cfg.get("dev_min_len", 3))
            mm = score(r, gold_path)
            if m is None or mm["arg_c_iou"]["f1"] > m["arg_c_iou"]["f1"]:
                m, recs, sel_tau = mm, r, t
        row = {
            "epoch": ep,
            "loss": tot / len(tr_dl),
            "type_acc": type_acc * 100,
            "arg_c_iou_f1": m["arg_c_iou"]["f1"],
            "argmax_arg_c_iou_f1": m_argmax["arg_c_iou"]["f1"],
            "sel_tau": sel_tau,
            "arg_i_iou_f1": m["arg_i_iou"]["f1"],
            "arg_c_iou_p": m["arg_c_iou"]["p"],
            "arg_c_iou_r": m["arg_c_iou"]["r"],
            "arg_c_em_f1": m["arg_c_exact"]["f1"],
            "arg_i_em_f1": m["arg_i_exact"]["f1"],
            "trigger_rougeL_f1": m["trigger_rougeL"]["f1"],
            "secs": round(time.time() - t0),
        }
        log.append(row)
        print(
            f"ep{ep:02d} loss {row['loss']:.3f} | type {row['type_acc']:.1f} | "
            f"ArgC-IoU {row['arg_c_iou_f1']:.2f} (P{row['arg_c_iou_p']:.1f}/R{row['arg_c_iou_r']:.1f}) | "
            f"ArgI-IoU {row['arg_i_iou_f1']:.2f} | raw {row['argmax_arg_c_iou_f1']:.2f} | t{sel_tau} | "
            f"RgL {row['trigger_rougeL_f1']:.2f} | {row['secs']}s",
            flush=True,
        )
        if row["arg_c_iou_f1"] > best["arg_c_iou_f1"]:
            best = dict(row)
            best["metrics"] = m
            write_jsonl(recs, os.path.join(run_dir, "dev_preds.jsonl"))
            if cfg.get("save_model", True):
                torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump({"config": cfg, "log": log, "best": best}, f, indent=2)
    print("\nBEST:", json.dumps({k: v for k, v in best.items() if k != "metrics"}, indent=2))
    return best


DEFAULTS = {
    "model_name": "microsoft/deberta-v3-large",
    "run_name": "h1_base",
    "out_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts", "runs"),
    "seed": 42,
    "epochs": 30,
    "batch_size": 8,
    "eval_batch_size": 16,
    "max_len": 640,
    "lr_encoder": 1e-5,
    "lr_head": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
    "max_grad_norm": 1.0,
    "w_role": 1.0,
    "w_aao": 0.5,
    "w_type": 0.5,
    "label_smoothing": 0.0,
    "type_teacher_min": 0.5,
    "use_lstm": False,
    "use_crf": False,
    "use_span_role": False,
    "w_span_role": 1.0,
    "llrd": 1.0,
    "dev_tau_grid": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "dev_min_len": 3,
    "dev_merge_gap": 0,
    "eval_split": "dev",
    "save_model": True,
    "data_dir": None,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(json.load(open(a.config)))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        cfg[k] = v
    cfg["out_dir"] = os.path.abspath(cfg["out_dir"])
    run(cfg)
