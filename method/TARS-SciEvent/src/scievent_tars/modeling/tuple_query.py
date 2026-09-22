"""TARS-H1: differentiable Agent-Action-Object tuple query.

Component representations are built from the *predicted* start/end
distributions, never from a hard gold span, so train and inference conditioning
are identical (no gold-trigger teacher forcing in the main path).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.schema import EVENT_TYPES, TRIGGER_COMPONENTS

_COMPONENT_ORDER = ("Agent", "Action", "PrimaryObject", "SecondaryObject")


class SoftComponent(nn.Module):
    """``g_c = MLP([g_start; g_end; presence_prob])`` with a learned NULL vector."""

    def __init__(self, hidden_size: int, out_dim: int, nullable: bool, dropout: float = 0.1):
        super().__init__()
        self.nullable = nullable
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.null_vector = nn.Parameter(torch.zeros(out_dim)) if nullable else None
        if self.null_vector is not None:
            nn.init.normal_(self.null_vector, std=0.02)

    def forward(
        self,
        start_logits: torch.Tensor,  # [B, K, S, W]
        end_logits: torch.Tensor,  # [B, K, S, W]
        word_states: torch.Tensor,  # [B, W, H]
        presence_logits: torch.Tensor | None,  # [B, K, S]
    ) -> torch.Tensor:
        p_start = torch.softmax(start_logits.float(), dim=-1).to(word_states.dtype)
        p_end = torch.softmax(end_logits.float(), dim=-1).to(word_states.dtype)
        g_start = torch.einsum("bksw,bwh->bksh", p_start, word_states)
        g_end = torch.einsum("bksw,bwh->bksh", p_end, word_states)

        if presence_logits is not None:
            presence = torch.sigmoid(presence_logits).unsqueeze(-1)
        else:
            presence = torch.ones_like(g_start[..., :1])

        g = self.mlp(torch.cat([g_start, g_end, presence], dim=-1))
        if self.null_vector is not None:
            g = presence * g + (1.0 - presence) * self.null_vector
        return g.mean(dim=2)  # pool the component's slots -> [B, K, out_dim]


class TupleQueryBuilder(nn.Module):
    """``q_tuple = LN(W_t [q_event; g_Agent; g_Action; g_PO; g_SO; e_type])``."""

    def __init__(self, hidden_size: int, component_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.components = nn.ModuleDict(
            {
                name: SoftComponent(
                    hidden_size, component_dim, nullable=(name != "Action"), dropout=dropout
                )
                for name in _COMPONENT_ORDER
            }
        )
        self.type_embed = nn.Embedding(len(EVENT_TYPES) + 1, component_dim)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.combine = nn.Sequential(
            nn.Linear(hidden_size + component_dim * (len(_COMPONENT_ORDER) + 1), hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        event_queries: torch.Tensor,  # [B, K, H]
        type_logits: torch.Tensor,  # [B, K, T+1]
        action: dict[str, torch.Tensor],
        components: dict[str, dict[str, torch.Tensor]],
        word_states: torch.Tensor,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = [event_queries]
        for name in _COMPONENT_ORDER:
            head_out = action if name == "Action" else components[name]
            pieces.append(
                self.components[name](
                    head_out["start_logits"],
                    head_out["end_logits"],
                    word_states,
                    head_out.get("presence_logits"),
                )
            )
        type_probs = torch.softmax(type_logits.float(), dim=-1).to(word_states.dtype)
        pieces.append(type_probs @ self.type_embed.weight.to(word_states.dtype))
        return self.norm(self.combine(torch.cat(pieces, dim=-1)))


assert set(_COMPONENT_ORDER) - {"Action"} == set(TRIGGER_COMPONENTS)
