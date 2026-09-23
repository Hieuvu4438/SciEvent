"""Stage 1 of HONE: generate candidate spans with the proposer (CARVE tagger).

Two modes.

  eval   Candidates for dev/test from proposers trained on the FULL train split
         (the three frozen CARVE seeds). These are what the verifier is
         validated and tested on.

  oof    Out-of-fold candidates for the TRAIN split. The proposer memorises its
         own training windows (final training loss ~0.002), so candidates it
         produces on windows it trained on contain almost no realistic errors
         and would teach a verifier nothing. Instead the train split is cut into
         K folds; for each fold a proposer is trained on the other K-1 folds
         (epoch selected on the real dev split, as for every model here) and
         decodes the held-out fold. Every train window thus receives candidates
         from a model that never saw it -- the same relationship dev/test
         windows have to the full-train proposers.

         Folds are window-level and stratified by event type, mirroring how the
         official split was built, so held-out windows keep sibling segments in
         the proposer's training data exactly as dev/test windows do.

         Fold checkpoints are deleted as soon as their candidates are written.

`--proposer simple` uses CARVE-simple (one merged 13-type BIO head, no type
conditioning; configs/proposer_simple.json) instead of the frozen CARVE recipe.
Its files carry the prefix `simple_`.

Usage:
  python3 scripts/propose.py eval --seed 42
  python3 scripts/propose.py oof  --fold 0 --seed 42
  python3 scripts/propose.py oof  --fold 0 --seed 42 --proposer simple
"""

import argparse
import collections
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from hone import candidates as C
from hone.data import AAO_LABEL2ID, AAO_TYPES, MERGED_LABEL2ID, MERGED_LABELS, ROLE_LABEL2ID, load_split
from hone.decode import posteriors
from hone.paths import default_data_dir, repo_root

MODEL = "microsoft/deberta-v3-large"
K_FOLDS = 5
FOLD_SEED = 0
ID2ROLE = {i: l for l, i in ROLE_LABEL2ID.items()}
ROOT = repo_root()
CARVE_RUNS = ROOT.parents[0] / "SciEvent-Next" / "artifacts" / "runs"   # frozen CARVE seeds
SIMPLE_RUNS = ROOT.parents[2] / "CARVE" / "runs"                          # CARVE-simple seeds
ID2MERGED = {i: l for l, i in MERGED_LABEL2ID.items()}
PROPOSERS = {
    "carve": {"prefix": "", "config": "proposer.json", "ckpt": lambda s: CARVE_RUNS / f"final_s{s}" / "best.pt"},
    # CARVE-simple full-train checkpoints; the CARVE repository calls these runs
    # proposer_s{seed} (they were carve_simple_s{seed} when this archive was written)
    "simple": {"prefix": "simple_", "config": "proposer_simple.json",
               "ckpt": lambda s: SIMPLE_RUNS / f"proposer_s{s}" / "best.pt"},
}


def merged_to_aao(p):
    """Single-head proposer: AAO/trigger spans come from the same merged tagger.
    Keep its own argmax decision, mapped into the AAO label space (non-AAO -> O),
    as a one-hot posterior so downstream argmax decoding is unchanged."""
    out = np.zeros((p.shape[0], len(AAO_LABEL2ID)), dtype=np.float32)
    for i, lid in enumerate(p.argmax(-1)):
        lab = ID2MERGED[int(lid)]
        keep = lab != "O" and lab.split("-", 1)[1] in AAO_TYPES
        out[i, AAO_LABEL2ID[lab] if keep else AAO_LABEL2ID["O"]] = 1.0
    return out


def rows_for(windows, rp, ap, tp, seed):
    single = rp[0].shape[-1] == len(MERGED_LABELS)
    rows = []
    for i, w in enumerate(windows):
        rows.append({
            "sent_id": w.sent_id,
            "cands": C.extract(rp[i], ID2MERGED if single else ID2ROLE, seed),
            "type_post": [round(float(x), 6) for x in tp[i]],
            "aao_post": (merged_to_aao(ap[i]) if single else np.round(ap[i], 5)).tolist(),
        })
    return rows


def mode_eval(seed, proposer="carve"):
    P = PROPOSERS[proposer]
    ck = P["ckpt"](seed)
    assert ck.exists(), f"missing frozen proposer checkpoint {ck}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    for split in ["dev", "test"]:
        w = load_split(split)
        rp, ap, tp = posteriors([str(ck)], MODEL, w, tok)
        out = ROOT / "data" / "cands" / f"{P['prefix']}{split}_s{seed}.jsonl"
        C.save(str(out), rows_for(w, rp, ap, tp, seed))
        n = sum(len(r["cands"]) for r in C.load(str(out)))
        print(f"[eval] {split} seed {seed}: {len(w)} windows, {n} candidates -> {out}")


def mode_insample(seed):
    """CONTROL for H4: the full-train proposer decoding its OWN training windows.

    These candidates are what a verifier would be trained on without
    cross-fitting. Used only for the H4 ablation, never by the shipped method.
    """
    ck = CARVE_RUNS / f"final_s{seed}" / "best.pt"
    tok = AutoTokenizer.from_pretrained(MODEL)
    w = load_split("train")
    rp, ap, tp = posteriors([str(ck)], MODEL, w, tok)
    out = ROOT / "data" / "cands" / f"insample_s{seed}.jsonl"
    C.save(str(out), rows_for(w, rp, ap, tp, seed))
    n = sum(len(r["cands"]) for r in C.load(str(out)))
    print(f"[insample] train seed {seed}: {len(w)} windows, {n} candidates -> {out}")


def build_folds():
    """Write K fold directories (idempotent). Window-level, stratified by type."""
    src = default_data_dir()
    raw = [json.loads(l) for l in open(os.path.join(src, "train.oneie.json")) if l.strip()]
    dev_lines = open(os.path.join(src, "dev.oneie.json")).read()
    by_type = collections.defaultdict(list)
    for r in raw:
        by_type[r["event_mentions"][0]["event_type"]].append(r)
    rng = random.Random(FOLD_SEED)
    fold_of = {}
    for t in sorted(by_type):
        rows = by_type[t][:]
        rng.shuffle(rows)
        for j, r in enumerate(rows):
            fold_of[r["sent_id"]] = j % K_FOLDS
    for k in range(K_FOLDS):
        d = ROOT / "data" / "folds" / str(k)
        if (d / "heldout.oneie.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "train.oneie.json", "w") as f:
            for r in raw:
                if fold_of[r["sent_id"]] != k:
                    f.write(json.dumps(r) + "\n")
        with open(d / "heldout.oneie.json", "w") as f:
            for r in raw:
                if fold_of[r["sent_id"]] == k:
                    f.write(json.dumps(r) + "\n")
        (d / "dev.oneie.json").write_text(dev_lines)
    return fold_of


def mode_oof(fold, seed, epochs=None, proposer="carve"):
    P = PROPOSERS[proposer]
    build_folds()
    fold_dir = ROOT / "data" / "folds" / str(fold)
    out = ROOT / "data" / "cands" / f"{P['prefix']}oof_k{fold}_s{seed}.jsonl"
    if out.exists():
        print(f"[oof] {out} already exists, skipping")
        return
    from hone.train import DEFAULTS, run
    cfg = dict(DEFAULTS)
    cfg.update(json.load(open(ROOT / "configs" / P["config"])))
    run_name = f"{P['prefix']}oof_k{fold}_s{seed}"
    cfg.update({"run_name": run_name, "seed": seed, "data_dir": str(fold_dir),
                "eval_split": "dev", "out_dir": str(ROOT / "runs"), "save_model": True})
    if epochs is not None:             # smoke tests only
        cfg["epochs"] = epochs
    run(cfg)
    torch.cuda.empty_cache()

    ck = ROOT / "runs" / run_name / "best.pt"
    held = load_split("heldout", str(fold_dir))
    tok = AutoTokenizer.from_pretrained(MODEL)
    rp, ap, tp = posteriors([str(ck)], MODEL, held, tok)
    C.save(str(out), rows_for(held, rp, ap, tp, seed))
    n = sum(len(r["cands"]) for r in C.load(str(out)))
    print(f"[oof] fold {fold} seed {seed}: {len(held)} held-out windows, {n} candidates -> {out}")
    ck.unlink()                       # keep disk usage flat; log.json is kept
    print(f"[oof] deleted {ck}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "oof", "insample"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--proposer", choices=list(PROPOSERS), default="carve")
    a = ap.parse_args()
    if a.mode == "eval":
        mode_eval(a.seed, a.proposer)
    elif a.mode == "insample":
        mode_insample(a.seed)
    else:
        assert a.fold is not None
        mode_oof(a.fold, a.seed, a.epochs, a.proposer)
