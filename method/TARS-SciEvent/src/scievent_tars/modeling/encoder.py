"""Backbone encoder + subword -> word mean pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def pool_pieces_to_words(
    piece_states: torch.Tensor,  # [B, P, H]
    piece_to_word: torch.Tensor,  # [B, P] long, -1 for specials/padding
    max_words: int,
) -> torch.Tensor:
    """``h_i = mean(piece states belonging to word i)``."""
    batch, _, hidden = piece_states.shape
    valid = piece_to_word >= 0
    index = piece_to_word.clamp(min=0)

    sums = piece_states.new_zeros((batch, max_words, hidden))
    counts = piece_states.new_zeros((batch, max_words, 1))

    contribution = piece_states * valid.unsqueeze(-1).to(piece_states.dtype)
    sums.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, hidden), contribution)
    counts.scatter_add_(
        1, index.unsqueeze(-1), valid.unsqueeze(-1).to(piece_states.dtype)
    )
    return sums / counts.clamp(min=1.0)


class WordEncoder(nn.Module):
    """Wraps the HF backbone and exposes word-level states."""

    def __init__(
        self,
        model_path: str,
        gradient_checkpointing: bool = True,
        dropout: float = 0.1,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        kwargs = {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.backbone = AutoModel.from_pretrained(model_path, config=config, **kwargs)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        self.hidden_size = int(config.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        piece_to_word: torch.Tensor,
        max_words: int,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        piece_states = out.last_hidden_state
        word_states = pool_pieces_to_words(piece_states, piece_to_word, max_words)
        return self.dropout(word_states)
