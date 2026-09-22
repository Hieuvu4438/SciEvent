"""Window-level event-set decoder.

``K_event`` learned queries attend to the word states and each slot predicts an
event type (or NONE) plus the Agent-Action-Object trigger structure. The model
never receives the gold event count or identity (``STOP_EVENT_01``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.schema import EVENT_TYPES, TRIGGER_COMPONENTS

NEG_INF = -1e4


class SpanSetHead(nn.Module):
    """Predicts ``n_slots`` (start, end) spans, optionally nullable, per event slot.

    Boundary logits are query-key dot products against the word states, which
    keeps the head cheap and length-agnostic.
    """

    def __init__(
        self,
        hidden_size: int,
        n_slots: int,
        nullable: bool,
        proj_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.nullable = nullable
        self.slot_embed = nn.Embedding(n_slots, hidden_size)
        nn.init.normal_(self.slot_embed.weight, std=0.02)

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_size), nn.GELU(), nn.Dropout(dropout)
        )
        self.start_key = nn.Linear(hidden_size, proj_size)
        self.end_key = nn.Linear(hidden_size, proj_size)
        self.start_query = nn.Linear(proj_size, proj_size)
        self.end_query = nn.Linear(proj_size, proj_size)
        self.scale = proj_size**-0.5
        self.presence = nn.Linear(proj_size, 1) if nullable else None

    def forward(
        self,
        event_queries: torch.Tensor,  # [B, K, H]
        word_states: torch.Tensor,  # [B, W, H]
        word_mask: torch.Tensor,  # [B, W] bool
    ) -> dict[str, torch.Tensor]:
        batch, k_event, _ = event_queries.shape
        slots = self.slot_embed.weight.unsqueeze(0).unsqueeze(0)  # [1,1,S,H]
        queries = event_queries.unsqueeze(2) + slots  # [B, K, S, H]
        q = self.query_proj(queries)  # [B, K, S, D]

        start_k = self.start_key(word_states)  # [B, W, D]
        end_k = self.end_key(word_states)

        start_logits = torch.einsum("bksd,bwd->bksw", self.start_query(q), start_k) * self.scale
        end_logits = torch.einsum("bksd,bwd->bksw", self.end_query(q), end_k) * self.scale

        pad = ~word_mask[:, None, None, :]
        start_logits = start_logits.masked_fill(pad, NEG_INF)
        end_logits = end_logits.masked_fill(pad, NEG_INF)

        out = {"start_logits": start_logits, "end_logits": end_logits}
        if self.presence is not None:
            out["presence_logits"] = self.presence(q).squeeze(-1)  # [B, K, S]
        return out


@dataclass
class EventSetOutput:
    queries: torch.Tensor  # [B, K, H] refined event-slot states
    type_logits: torch.Tensor  # [B, K, num_types + 1]
    action: dict[str, torch.Tensor]
    components: dict[str, dict[str, torch.Tensor]]


class EventSetDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        k_event: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        component_slots: dict[str, int] | None = None,
        proj_size: int = 256,
    ):
        super().__init__()
        self.k_event = k_event
        self.query_embed = nn.Embedding(k_event, hidden_size)
        nn.init.normal_(self.query_embed.weight, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.query_norm = nn.LayerNorm(hidden_size)

        self.type_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(EVENT_TYPES) + 1),
        )
        self.action_head = SpanSetHead(
            hidden_size, n_slots=1, nullable=False, proj_size=proj_size, dropout=dropout
        )
        slots = component_slots or {c: 1 for c in TRIGGER_COMPONENTS}
        self.component_heads = nn.ModuleDict(
            {
                c: SpanSetHead(
                    hidden_size,
                    n_slots=slots.get(c, 1),
                    nullable=True,
                    proj_size=proj_size,
                    dropout=dropout,
                )
                for c in TRIGGER_COMPONENTS
            }
        )

    def forward(
        self,
        word_states: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> EventSetOutput:
        batch = word_states.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        refined = self.decoder(
            tgt=queries,
            memory=word_states,
            memory_key_padding_mask=~word_mask,
        )
        refined = self.query_norm(refined)

        return EventSetOutput(
            queries=refined,
            type_logits=self.type_head(refined),
            action=self.action_head(refined, word_states, word_mask),
            components={
                name: head(refined, word_states, word_mask)
                for name, head in self.component_heads.items()
            },
        )
