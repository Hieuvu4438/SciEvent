"""CARVE model: a multi-task word-level span tagger.

Architecture (H1, "SASR" = Segment-Aware Span-Role tagger):

    shared encoder (DeBERTa-v3-large by default)
        -> first-subword pooling to word-level states
        -> ROLE head : BIO over the 9 scored semantic roles
        -> AAO  head : BIO over <Agent, Action, PrimaryObject, SecondaryObject>
        -> TYPE head : 4-way window event-type classification (attention pooled)

Rationale: the official metric is trigger-insensitive but event-type sensitive,
and the scored arguments are long, near-contiguous, mutually non-overlapping
clause spans. That is a span-segmentation problem, not an entity-mention
problem, so a discriminative word tagger dominates generative copying
(cf. DEGREE recall of 19.1 on this benchmark).

Event-type conditioning is applied by adding a learned event-type embedding to
the word states before the ROLE head. During training we feed the *gold* type
with probability `type_teacher_p` and the model's own argmax otherwise
(scheduled sampling), so train and inference distributions stay aligned.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from carve.crf import CRF


class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=-1).unsqueeze(-1)
        return (x * w).sum(1)


class CarveModel(nn.Module):
    def __init__(
        self,
        model_name,
        n_role_labels,
        n_aao_labels,
        n_event_types,
        dropout=0.1,
        use_crf=False,
        use_span_role=False,
        use_type_cond=True,
        single_head=False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # NOTE: several checkpoints (e.g. deberta-v3-large) store fp16 weights and
        # transformers>=5 honours that dtype, which makes AdamW produce NaN on the
        # very first step. Master weights must be fp32; bf16 comes from autocast.
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        h = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPool(h)
        self.type_head = nn.Linear(h, n_event_types)
        self.type_emb = nn.Embedding(n_event_types, h)
        nn.init.zeros_(self.type_emb.weight)
        self.use_type_cond = use_type_cond
        self.single_head = single_head
        # Under `single_head` the ROLE head carries the merged label space and the
        # AAO head does not exist; this ablates the two-disjoint-heads decision.
        self.role_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_role_labels))
        if not single_head:
            self.aao_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_aao_labels))
        self.use_span_role = use_span_role
        if use_span_role:
            # span-level role classifier: [h_start ; h_end ; mean(h)] -> role
            self.span_role_head = nn.Sequential(
                nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_role_labels // 2)
            )
        self.use_crf = use_crf
        if use_crf:
            self.role_crf = CRF(n_role_labels)
            self.aao_crf = CRF(n_aao_labels)

    def word_states(self, input_ids, attention_mask, word_index, word_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # gather the first sub-token of every word
        idx = word_index.unsqueeze(-1).expand(-1, -1, out.size(-1))
        w = torch.gather(out, 1, idx)
        w = w * word_mask.unsqueeze(-1)
        return self.dropout(w)

    def forward(self, input_ids, attention_mask, word_index, word_mask, gold_type=None, type_teacher_p=1.0, return_states=False):
        w = self.word_states(input_ids, attention_mask, word_index, word_mask)
        pooled = self.pool(w, word_mask.bool())
        type_logits = self.type_head(pooled)

        if gold_type is not None:
            pred = type_logits.argmax(-1)
            keep = (torch.rand_like(pred, dtype=torch.float) < type_teacher_p)
            cond = torch.where(keep, gold_type, pred)
        else:
            cond = type_logits.argmax(-1)
        w_cond = w + self.type_emb(cond).unsqueeze(1) if self.use_type_cond else w

        aao_logits = None if self.single_head else self.aao_head(w)
        out = (type_logits, self.role_head(w_cond), aao_logits)
        return out + (w_cond,) if return_states else out

    def span_roles(self, w_cond, spans):
        """Classify a list of (batch_index, start, end) spans into role types.

        The BIO head decides a role per *token*; this head sees the whole span at
        once. It targets the Arg-I vs Arg-C gap, which is pure role confusion:
        the span is right and the label is wrong.
        """
        if not spans:
            return None
        reps = []
        for b, s, e in spans:
            h = w_cond[b, s:e]
            reps.append(torch.cat([w_cond[b, s], w_cond[b, e - 1], h.mean(0)], dim=-1))
        return self.span_role_head(torch.stack(reps))
