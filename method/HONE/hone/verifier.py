"""Stage 2 of HONE: the learned verifier.

For one candidate span the verifier sees

    event type: <predicted window type> .  w_1 ... <a> w_s ... w_{e-1} </a> ... w_n

i.e. the whole window with the candidate wrapped in marker tokens (entity-marker
encoding, Zhong & Chen 2021), plus the proposer's evidence for that span as a
small feature vector. It outputs one of 10 classes: `reject`, or one of the 9
scored roles. So it does two jobs CARVE's confidence threshold could not:

  * decide keep/drop from the *content* of the span in context, instead of from
    a mean token posterior that carries no signal for short spans;
  * re-assign the role, which a threshold cannot do at all.

Training candidates are out-of-fold (see scripts/propose.py), so the verifier
learns from the proposer's real mistakes.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel

from hone.candidates import LABEL2ID
from hone.data import EVENT_TYPES, ROLE_TYPES

MARK_OPEN, MARK_CLOSE = "<a>", "</a>"
N_FEATS = len(ROLE_TYPES) + 8


def features(c, n_words):
    """Proposer evidence for one candidate, all roughly in [0, 1]."""
    length = c["e"] - c["s"]
    return (list(c["role_mass"]) + [
        c["o_mass"],
        c["conf"],
        c.get("conf_max", c["conf"]),
        c.get("n_agree", 1) / 3.0,
        math.log1p(length) / math.log1p(60),
        c["s"] / max(1, n_words),
        c["e"] / max(1, n_words),
        (c.get("n_parts", 1) - 1) / 3.0,
    ])


def build_examples(windows_by_id, rows, with_labels=True):
    """Flatten per-window candidate rows into verifier examples."""
    ex = []
    for r in rows:
        w = windows_by_id[r["sent_id"]]
        etype = EVENT_TYPES[int(np.argmax(r["type_post"]))]
        for j, c in enumerate(r["cands"]):
            ex.append({
                "sent_id": r["sent_id"], "cand_idx": j, "tokens": w.tokens, "etype": etype,
                "s": c["s"], "e": c["e"], "feats": features(c, len(w.tokens)),
                "label": LABEL2ID[c["label"]] if with_labels else -100,
            })
    return ex


class VerifierDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=320):
        self.ex = examples
        self.tok = tokenizer
        self.max_len = max_len
        self.open_id = tokenizer.convert_tokens_to_ids(MARK_OPEN)
        self.close_id = tokenizer.convert_tokens_to_ids(MARK_CLOSE)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        x = self.ex[i]
        t = x["tokens"]
        words = (["event", "type:", x["etype"], "."] + t[:x["s"]] + [MARK_OPEN] +
                 t[x["s"]:x["e"]] + [MARK_CLOSE] + t[x["e"]:])
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=self.max_len)
        ids = enc["input_ids"]
        # markers are single special tokens; locate them directly
        o = ids.index(self.open_id) if self.open_id in ids else 0
        c = ids.index(self.close_id) if self.close_id in ids else 0
        return {"input_ids": ids, "open": o, "close": c,
                "feats": x["feats"], "label": x["label"], "idx": i}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        att[i, :len(b["input_ids"])] = 1
    return {
        "input_ids": ids, "attention_mask": att,
        "open": torch.tensor([b["open"] for b in batch]),
        "close": torch.tensor([b["close"] for b in batch]),
        "feats": torch.tensor([b["feats"] for b in batch], dtype=torch.float),
        "label": torch.tensor([b["label"] for b in batch]),
        "idx": torch.tensor([b["idx"] for b in batch]),
    }


class Verifier(nn.Module):
    def __init__(self, model_name, vocab_size, n_labels=len(LABEL2ID), dropout=0.1, use_feats=True,
                 n_feats=N_FEATS):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # fp32 master weights: several checkpoints ship fp16 and transformers>=5
        # honours that, which NaNs AdamW on the first step.
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.encoder.resize_token_embeddings(vocab_size)
        h = self.config.hidden_size
        self.use_feats = use_feats
        fdim = 64 if use_feats else 0
        # n_feats < N_FEATS loads checkpoints trained before a feature was appended
        # (features are only ever appended, so the leading ones keep their meaning)
        self.n_feats = n_feats
        if use_feats:
            self.feat = nn.Sequential(nn.Linear(n_feats, fdim), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(3 * h + fdim, h), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(h, n_labels))

    def forward(self, input_ids, attention_mask, open_idx, close_idx, feats):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ar = torch.arange(hs.size(0), device=hs.device)
        parts = [hs[:, 0], hs[ar, open_idx], hs[ar, close_idx]]
        if self.use_feats:
            parts.append(self.feat(feats[:, :self.n_feats]))
        return self.head(self.dropout(torch.cat(parts, dim=-1)))
