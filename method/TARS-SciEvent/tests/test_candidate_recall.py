"""Candidate span proposal contract.

* Gold spans injected at training time are always present (so a proposal miss
  can never be mis-attributed to the role decoder).
* The ``max_span_width`` chosen from TRAIN keeps the achievable recall ceiling
  above the STOP_MODEL_02 gate of 98% on dev.
"""

import math

import torch

from conftest import DATA_DIR

from scievent_tars.data.reader import read_split, train_statistics
from scievent_tars.modeling.span_candidates import SpanCandidateModule


def _resolved_max_span_width() -> int:
    stats = train_statistics(read_split(DATA_DIR / "train.json"))
    return max(16, min(64, int(math.ceil(stats["semantic_arg_width"]["p99_5"]))))


def test_width_cap_recall_ceiling_clears_the_gate():
    cap = _resolved_max_span_width()
    for split, threshold in (("train", 0.99), ("dev", 0.98)):
        spans = [
            a.span
            for ex in read_split(DATA_DIR / f"{split}.json")
            for ev in ex.events
            for a in ev.arguments
        ]
        reachable = sum(1 for s in spans if s.width <= cap)
        recall = reachable / len(spans)
        assert recall >= threshold, (
            f"{split}: width cap {cap} caps candidate recall at {recall:.4f} "
            f"(< {threshold})"
        )


def test_gold_spans_are_always_injected_during_training():
    torch.manual_seed(0)
    hidden, num_words = 32, 60
    module = SpanCandidateModule(
        hidden, max_span_width=40, k_start=8, k_end=8, k_span=16, out_dim=32
    )
    word_states = torch.randn(1, num_words, hidden)
    word_mask = torch.ones(1, num_words, dtype=torch.bool)
    num_words_t = torch.tensor([num_words])

    gold = [(3, 9), (20, 41), (50, 58)]
    out = module(word_states, word_mask, num_words_t, forced_spans=[gold])
    for span in gold:
        assert span in out.index_of[0], f"gold span {span} missing from candidates"
        idx = out.index_of[0][span]
        assert tuple(out.spans[0, idx].tolist()) == span
        assert bool(out.mask[0, idx])


def test_candidates_respect_width_and_bounds():
    torch.manual_seed(0)
    hidden, num_words, cap = 32, 45, 12
    module = SpanCandidateModule(
        hidden, max_span_width=cap, k_start=16, k_end=16, k_span=64, out_dim=32
    )
    out = module(
        torch.randn(1, num_words, hidden),
        torch.ones(1, num_words, dtype=torch.bool),
        torch.tensor([num_words]),
        forced_spans=None,
    )
    for span, valid in zip(out.spans[0].tolist(), out.mask[0].tolist()):
        if not valid:
            continue
        start, end = span
        assert 0 <= start < end <= num_words
        assert end - start <= cap


def test_padded_batch_keeps_candidates_inside_each_window():
    torch.manual_seed(0)
    hidden = 32
    module = SpanCandidateModule(
        hidden, max_span_width=10, k_start=8, k_end=8, k_span=32, out_dim=32
    )
    word_mask = torch.zeros(2, 40, dtype=torch.bool)
    word_mask[0, :40] = True
    word_mask[1, :12] = True
    out = module(
        torch.randn(2, 40, hidden), word_mask, torch.tensor([40, 12]), forced_spans=None
    )
    for span, valid in zip(out.spans[1].tolist(), out.mask[1].tolist()):
        if valid:
            assert span[1] <= 12, f"candidate {span} leaks into padding"
