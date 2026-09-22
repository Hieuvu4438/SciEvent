"""Per-event, per-role argument set decoder.

``q_arg[k, r, m] = MLP([cond_k ; p_role[r] ; E_slot[m]])`` is scored against
every candidate span plus a NULL candidate. Role slots are matched to the gold
span set of that role with a per-role Hungarian assignment, which gives
permutation-invariant support for multiple arguments of the same role.

``cond_k`` is the raw event-slot state in H0 and the compositional tuple query
in H1+.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.schema import SEMANTIC_ROLES

NEG_INF = -1e4


@dataclass
class RoleSlotLayout:
    """Flat indexing over ``sum_r K_arg_role[r]`` role slots."""

    counts: dict[str, int]
    role_index: list[int]  # flat slot -> role id
    slot_index: list[int]  # flat slot -> within-role slot id
    offsets: dict[str, tuple[int, int]]  # role -> (start, end) in the flat axis

    @classmethod
    def build(cls, counts: dict[str, int]) -> "RoleSlotLayout":
        role_index: list[int] = []
        slot_index: list[int] = []
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for rid, role in enumerate(SEMANTIC_ROLES):
            n = int(counts[role])
            if n < 1:
                raise ValueError(f"role {role!r} needs at least one slot")
            offsets[role] = (cursor, cursor + n)
            role_index.extend([rid] * n)
            slot_index.extend(range(n))
            cursor += n
        return cls(counts=dict(counts), role_index=role_index, slot_index=slot_index,
                   offsets=offsets)

    @property
    def total(self) -> int:
        return len(self.role_index)

    @property
    def max_slots(self) -> int:
        return max(self.counts.values())


class ArgumentSetDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        candidate_dim: int,
        layout: RoleSlotLayout,
        proj_size: int = 512,
        dropout: float = 0.1,
        use_action_distance: bool = True,
        max_sentence_distance: int = 8,
    ):
        super().__init__()
        self.layout = layout
        self.use_action_distance = use_action_distance
        self.max_sentence_distance = max_sentence_distance

        self.role_slot_embed = nn.Embedding(layout.max_slots, hidden_size)
        nn.init.normal_(self.role_slot_embed.weight, std=0.02)

        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_size, proj_size),
            nn.LayerNorm(proj_size),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_dim, proj_size), nn.LayerNorm(proj_size)
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.null_vector = nn.Parameter(torch.zeros(proj_size))
        nn.init.normal_(self.null_vector, std=0.02)
        self.scale = proj_size**-0.5

        if use_action_distance:
            self.distance_embed = nn.Embedding(2 * max_sentence_distance + 1, 1)
            nn.init.zeros_(self.distance_embed.weight)

        self.register_buffer(
            "role_index_buf", torch.tensor(layout.role_index, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "slot_index_buf", torch.tensor(layout.slot_index, dtype=torch.long), persistent=False
        )

    def _distance_bias(
        self,
        action_start_logits: torch.Tensor,  # [B, K, 1, W]
        candidate_spans: torch.Tensor,  # [B, C, 2]
        word_sentence_ids: torch.Tensor,  # [B, W]
    ) -> torch.Tensor:
        with torch.no_grad():
            action_word = action_start_logits[:, :, 0, :].argmax(dim=-1)  # [B, K]
            action_sent = word_sentence_ids.gather(1, action_word)  # [B, K]
            cand_sent = word_sentence_ids.gather(1, candidate_spans[:, :, 0])  # [B, C]
            delta = cand_sent.unsqueeze(1) - action_sent.unsqueeze(2)  # [B, K, C]
            delta = delta.clamp(-self.max_sentence_distance, self.max_sentence_distance)
            delta = delta + self.max_sentence_distance
        return self.distance_embed(delta).squeeze(-1)  # [B, K, C]

    def forward(
        self,
        condition: torch.Tensor,  # [B, K, H] event-slot state or tuple query
        role_vectors: torch.Tensor,  # [num_roles, H]
        candidate_features: torch.Tensor,  # [B, C, D]
        candidate_mask: torch.Tensor,  # [B, C]
        candidate_spans: torch.Tensor,  # [B, C, 2]
        word_sentence_ids: torch.Tensor,  # [B, W]
        action_start_logits: torch.Tensor | None,  # [B, K, 1, W]
        null_bias: float = 0.0,
    ) -> torch.Tensor:
        """Returns logits ``[B, K, M, C + 1]`` where index ``C`` is NULL."""
        batch, k_event, hidden = condition.shape
        total_slots = self.layout.total

        roles = role_vectors.to(condition.dtype)[self.role_index_buf]  # [M, H]
        slots = self.role_slot_embed(self.slot_index_buf).to(condition.dtype)  # [M, H]
        query_parts = torch.cat([roles, slots], dim=-1)  # [M, 2H]
        query_parts = query_parts.unsqueeze(0).unsqueeze(0).expand(batch, k_event, -1, -1)
        cond = condition.unsqueeze(2).expand(-1, -1, total_slots, -1)
        queries = self.query_mlp(torch.cat([cond, query_parts], dim=-1))  # [B,K,M,P]

        keys = self.candidate_proj(candidate_features)  # [B, C, P]
        logits = torch.einsum("bkmp,bcp->bkmc", queries, keys) * self.scale + self.bias

        if self.use_action_distance and action_start_logits is not None:
            logits = logits + self._distance_bias(
                action_start_logits, candidate_spans, word_sentence_ids
            ).unsqueeze(2)

        logits = logits.masked_fill(~candidate_mask[:, None, None, :], NEG_INF)

        null_logits = (
            torch.einsum("bkmp,p->bkm", queries, self.null_vector.to(queries.dtype))
            * self.scale
            + null_bias
        )
        return torch.cat([logits, null_logits.unsqueeze(-1)], dim=-1)
