"""Global candidate span proposal.

Boundary heads score start words and (inclusive) end words independently; valid
pairs under ``max_span_width`` are ranked by boundary score and the top
``K_span`` survive. During training every gold semantic span is injected into
the candidate set so a proposal miss can never be mistaken for a role-decoder
error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

NEG_INF = -1e4


@dataclass
class CandidateSet:
    spans: torch.Tensor  # [B, C, 2] long, end-exclusive word coords
    mask: torch.Tensor  # [B, C] bool
    features: torch.Tensor  # [B, C, D]
    scores: torch.Tensor  # [B, C] boundary score (start + end logit)
    start_logits: torch.Tensor  # [B, W]
    end_logits: torch.Tensor  # [B, W] over inclusive end words
    index_of: list[dict[tuple[int, int], int]]  # per-example span -> candidate idx


class SpanCandidateModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_span_width: int,
        k_start: int = 64,
        k_end: int = 64,
        k_span: int = 512,
        width_embed_dim: int = 64,
        out_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_span_width = max_span_width
        self.k_start = k_start
        self.k_end = k_end
        self.k_span = k_span

        self.start_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )
        self.end_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )
        self.attn_score = nn.Linear(hidden_size, 1)
        self.width_embed = nn.Embedding(max_span_width + 1, width_embed_dim)
        self.project = nn.Sequential(
            nn.Linear(hidden_size * 3 + width_embed_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    # -- proposal ---------------------------------------------------------------

    def _propose(
        self,
        start_logits: torch.Tensor,  # [W]
        end_logits: torch.Tensor,  # [W]
        num_words: int,
        forced: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        k_start = min(self.k_start, num_words)
        k_end = min(self.k_end, num_words)
        top_starts = torch.topk(start_logits[:num_words], k_start).indices
        top_ends = torch.topk(end_logits[:num_words], k_end).indices

        grid_start = top_starts.unsqueeze(1).expand(k_start, k_end)
        grid_end = top_ends.unsqueeze(0).expand(k_start, k_end)
        width = grid_end - grid_start + 1
        valid = (width >= 1) & (width <= self.max_span_width)
        pair_score = start_logits[grid_start] + end_logits[grid_end]
        pair_score = pair_score.masked_fill(~valid, float("-inf"))

        keep = min(self.k_span, int(valid.sum()))
        chosen: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for span in forced:  # gold injection keeps priority
            if span not in seen:
                seen.add(span)
                chosen.append(span)

        if keep > 0:
            flat = torch.topk(pair_score.flatten(), keep).indices
            ranked_start = grid_start.flatten()[flat].tolist()
            ranked_end = grid_end.flatten()[flat].tolist()
            for s, e_incl in zip(ranked_start, ranked_end):
                if len(chosen) >= self.k_span:
                    break
                span = (s, e_incl + 1)
                if span in seen:
                    continue
                seen.add(span)
                chosen.append(span)

        if not chosen:
            chosen = [(0, min(1, num_words))]
        return chosen

    # -- representation ---------------------------------------------------------

    def _featurize(
        self,
        word_states: torch.Tensor,  # [W, H]
        spans: torch.Tensor,  # [C, 2]
        attn_logits: torch.Tensor,  # [W]
        num_words: int,
    ) -> torch.Tensor:
        starts = spans[:, 0]
        ends_incl = spans[:, 1] - 1

        positions = torch.arange(num_words, device=word_states.device)
        inside = (positions[None, :] >= starts[:, None]) & (
            positions[None, :] <= ends_incl[:, None]
        )
        weights = attn_logits[None, :num_words].masked_fill(~inside, NEG_INF)
        weights = torch.softmax(weights.float(), dim=-1).to(word_states.dtype)
        pooled = weights @ word_states[:num_words]

        widths = (spans[:, 1] - spans[:, 0]).clamp(min=1, max=self.max_span_width)
        return self.project(
            torch.cat(
                [
                    word_states[starts],
                    word_states[ends_incl],
                    pooled,
                    self.width_embed(widths),
                ],
                dim=-1,
            )
        )

    def forward(
        self,
        word_states: torch.Tensor,  # [B, W, H]
        word_mask: torch.Tensor,  # [B, W]
        num_words: torch.Tensor,  # [B]
        forced_spans: list[list[tuple[int, int]]] | None = None,
        only_forced: bool = False,
    ) -> CandidateSet:
        batch, max_words, _ = word_states.shape
        device = word_states.device

        start_logits = self.start_head(word_states).squeeze(-1)
        end_logits = self.end_head(word_states).squeeze(-1)
        start_logits = start_logits.masked_fill(~word_mask, NEG_INF)
        end_logits = end_logits.masked_fill(~word_mask, NEG_INF)
        attn_logits = self.attn_score(word_states).squeeze(-1)

        with torch.no_grad():
            detached_start = start_logits.detach().float()
            detached_end = end_logits.detach().float()

        per_example: list[list[tuple[int, int]]] = []
        for b in range(batch):
            forced = forced_spans[b] if forced_spans is not None else []
            n = int(num_words[b])
            forced = [(s, e) for (s, e) in forced if 0 <= s < e <= n]
            if only_forced:
                # O2 oracle: perfect boundaries, role classification only.
                per_example.append(list(dict.fromkeys(forced)) or [(0, min(1, n))])
            else:
                per_example.append(
                    self._propose(detached_start[b], detached_end[b], n, forced)
                )

        max_candidates = max(len(c) for c in per_example)
        spans_tensor = torch.zeros((batch, max_candidates, 2), dtype=torch.long, device=device)
        mask = torch.zeros((batch, max_candidates), dtype=torch.bool, device=device)
        features = word_states.new_zeros((batch, max_candidates, self.out_dim))
        scores = word_states.new_full((batch, max_candidates), NEG_INF)
        index_of: list[dict[tuple[int, int], int]] = []

        for b, cands in enumerate(per_example):
            c = len(cands)
            span_t = torch.tensor(cands, dtype=torch.long, device=device)
            spans_tensor[b, :c] = span_t
            mask[b, :c] = True
            features[b, :c] = self._featurize(
                word_states[b], span_t, attn_logits[b], int(num_words[b])
            )
            scores[b, :c] = start_logits[b][span_t[:, 0]] + end_logits[b][span_t[:, 1] - 1]
            index_of.append({span: i for i, span in enumerate(cands)})

        return CandidateSet(
            spans=spans_tensor,
            mask=mask,
            features=features,
            scores=scores,
            start_logits=start_logits,
            end_logits=end_logits,
            index_of=index_of,
        )
