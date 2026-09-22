"""Batching for window-level examples.

The collator emits dense piece tensors plus the ``piece -> word`` assignment
used for mean pooling, and carries the gold structure through as Python objects
because the Hungarian matchers operate per example.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .schema import WindowExample
from .tokenizer_map import EncodedWindow, encode_window


@dataclass(slots=True)
class Batch:
    input_ids: torch.Tensor  # [B, P]
    attention_mask: torch.Tensor  # [B, P]
    piece_to_word: torch.Tensor  # [B, P] long, -1 for specials/padding
    word_mask: torch.Tensor  # [B, W] bool
    word_sentence_ids: torch.Tensor  # [B, W] long
    num_words: torch.Tensor  # [B] long
    examples: list[WindowExample]
    wnd_ids: list[str]

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            piece_to_word=self.piece_to_word.to(device, non_blocking=True),
            word_mask=self.word_mask.to(device, non_blocking=True),
            word_sentence_ids=self.word_sentence_ids.to(device, non_blocking=True),
            num_words=self.num_words.to(device, non_blocking=True),
            examples=self.examples,
            wnd_ids=self.wnd_ids,
        )

    def __len__(self) -> int:
        return len(self.examples)


class WindowDataset(Dataset):
    """Pre-encodes every window once; the corpus is small enough to hold in RAM."""

    def __init__(self, examples: list[WindowExample], tokenizer, max_length: int):
        self.examples = examples
        self.encoded: list[EncodedWindow] = [
            encode_window(ex, tokenizer, max_length) for ex in examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[WindowExample, EncodedWindow]:
        return self.examples[idx], self.encoded[idx]

    def piece_lengths(self) -> list[int]:
        return [e.num_pieces for e in self.encoded]


def collate(
    items: list[tuple[WindowExample, EncodedWindow]], pad_token_id: int
) -> Batch:
    batch_size = len(items)
    max_pieces = max(e.num_pieces for _, e in items)
    max_words = max(e.num_words for _, e in items)

    input_ids = torch.full((batch_size, max_pieces), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_pieces), dtype=torch.long)
    piece_to_word = torch.full((batch_size, max_pieces), -1, dtype=torch.long)
    word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
    word_sentence_ids = torch.zeros((batch_size, max_words), dtype=torch.long)
    num_words = torch.zeros((batch_size,), dtype=torch.long)

    for b, (_, enc) in enumerate(items):
        p = enc.num_pieces
        input_ids[b, :p] = torch.tensor(enc.input_ids, dtype=torch.long)
        attention_mask[b, :p] = torch.tensor(enc.attention_mask, dtype=torch.long)
        for word_idx, (first, last) in enumerate(enc.word_piece_spans):
            piece_to_word[b, first:last] = word_idx
        w = enc.num_words
        word_mask[b, :w] = True
        word_sentence_ids[b, :w] = torch.tensor(enc.word_sentence_ids, dtype=torch.long)
        num_words[b] = w

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        piece_to_word=piece_to_word,
        word_mask=word_mask,
        word_sentence_ids=word_sentence_ids,
        num_words=num_words,
        examples=[ex for ex, _ in items],
        wnd_ids=[ex.wnd_id for ex, _ in items],
    )


class LengthBucketSampler(torch.utils.data.Sampler[list[int]]):
    """Groups windows of similar piece length to cut padding waste.

    Shuffles within a shuffled pool of ``bucket_size * batch_size`` examples so
    batches stay stochastic while remaining length-homogeneous.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_multiplier: int = 32,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_size = max(batch_size * bucket_multiplier, batch_size)
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        n = len(self.lengths)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=generator).tolist()
        else:
            order = list(range(n))

        batches: list[list[int]] = []
        for pool_start in range(0, n, self.pool_size):
            pool = order[pool_start : pool_start + self.pool_size]
            pool.sort(key=lambda i: self.lengths[i])
            for bs in range(0, len(pool), self.batch_size):
                chunk = pool[bs : bs + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch + 10_000)
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        yield from batches

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
