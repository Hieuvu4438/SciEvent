"""TARS-H4: lightweight rhetorical sentence hierarchy.

OFF by default. It is a hypothesis, not an assumed improvement, and it is only
switched on after the H1-H3 diagnostics motivate it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SentenceHierarchy(nn.Module):
    """``s_m = mean(h_i in sentence m)``; ``h'_i = LN(h_i + W_s S'[sent(i)])``."""

    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.project = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        word_states: torch.Tensor,  # [B, W, H]
        word_mask: torch.Tensor,  # [B, W] bool
        word_sentence_ids: torch.Tensor,  # [B, W] long
    ) -> torch.Tensor:
        batch, num_words, hidden = word_states.shape
        num_sentences = int(word_sentence_ids.masked_fill(~word_mask, 0).max().item()) + 1

        index = word_sentence_ids.clamp(min=0) * word_mask.long()
        sums = word_states.new_zeros((batch, num_sentences, hidden))
        counts = word_states.new_zeros((batch, num_sentences, 1))
        masked = word_states * word_mask.unsqueeze(-1).to(word_states.dtype)
        sums.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, hidden), masked)
        counts.scatter_add_(
            1, index.unsqueeze(-1), word_mask.unsqueeze(-1).to(word_states.dtype)
        )
        sentence_states = sums / counts.clamp(min=1.0)
        sentence_pad = counts.squeeze(-1) <= 0

        refined = self.encoder(sentence_states, src_key_padding_mask=sentence_pad)
        refined = torch.nan_to_num(refined)

        gathered = refined.gather(
            1, index.unsqueeze(-1).expand(-1, -1, hidden)
        )
        return self.norm(word_states + self.project(gathered))
