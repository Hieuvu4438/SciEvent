"""Word <-> subword mapping.

The model scores *word-level* states obtained by pooling each word's subword
pieces, so every gold span round-trips exactly to its original word indices.
Truncation that would drop a gold span is a hard error (``STOP_MODEL_01``).
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .schema import Span, WindowExample


class GoldSpanTruncated(RuntimeError):
    """STOP_MODEL_01: a gold span does not survive tokenization."""


@dataclass(slots=True)
class EncodedWindow:
    wnd_id: str
    input_ids: list[int]
    attention_mask: list[int]
    #: ``word_piece_spans[i] == (first_piece, last_piece_exclusive)`` for word i.
    word_piece_spans: list[tuple[int, int]]
    num_words: int
    #: 0-based sentence index per word.
    word_sentence_ids: list[int]

    @property
    def num_pieces(self) -> int:
        return len(self.input_ids)


def load_tokenizer(path: str) -> PreTrainedTokenizerFast:
    tok = AutoTokenizer.from_pretrained(path)
    if not tok.is_fast:
        raise ValueError(
            f"{path}: a fast tokenizer is required for word_ids() alignment"
        )
    return tok


def encode_window(
    example: WindowExample,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
) -> EncodedWindow:
    """Tokenize a window and build the word -> piece-range map.

    Words whose pieces fall entirely beyond ``max_length`` are dropped from the
    map; if any gold span touches such a word we raise rather than silently
    training on a truncated target.
    """
    enc = tokenizer(
        example.words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    word_ids = enc.word_ids()

    spans: list[tuple[int, int] | None] = [None] * example.num_words
    for piece_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        current = spans[word_idx]
        if current is None:
            spans[word_idx] = (piece_idx, piece_idx + 1)
        else:
            spans[word_idx] = (current[0], piece_idx + 1)

    missing = [i for i, s in enumerate(spans) if s is None]
    if missing:
        first_lost = missing[0]
        for ev in example.events:
            covered = [ev.action_span, *ev.agent_spans, *ev.primary_object_spans,
                       *ev.secondary_object_spans, *(a.span for a in ev.arguments)]
            for sp in covered:
                if sp.end > first_lost:
                    raise GoldSpanTruncated(
                        f"{example.wnd_id}: max_length={max_length} truncates word "
                        f"{first_lost} but gold span {sp.as_tuple()} extends to "
                        f"{sp.end}. Increase max_length or implement an explicit "
                        f"chunking strategy."
                    )
        # No gold span is affected: keep only the words that survived.
        spans = spans[:first_lost]

    resolved = [s for s in spans if s is not None]
    sentence_ids = example.sentence_id_of_word()[: len(resolved)]

    return EncodedWindow(
        wnd_id=example.wnd_id,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        word_piece_spans=resolved,
        num_words=len(resolved),
        word_sentence_ids=sentence_ids,
    )


def roundtrip_span(encoded: EncodedWindow, span: Span) -> Span:
    """Map a word span to pieces and back, to prove the mapping is lossless."""
    if span.end > encoded.num_words:
        raise GoldSpanTruncated(
            f"{encoded.wnd_id}: span {span.as_tuple()} beyond {encoded.num_words} "
            f"mapped words"
        )
    first_piece = encoded.word_piece_spans[span.start][0]
    last_piece = encoded.word_piece_spans[span.end - 1][1]

    start_word = next(
        i for i, (a, _) in enumerate(encoded.word_piece_spans) if a == first_piece
    )
    end_word = (
        next(i for i, (_, b) in enumerate(encoded.word_piece_spans) if b == last_piece)
        + 1
    )
    return Span(start_word, end_word)


def audit_lengths(
    examples: list[WindowExample],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
) -> dict:
    """Report the subword-length distribution and prove no gold span is lost."""
    lengths: list[int] = []
    for ex in examples:
        enc = tokenizer(
            ex.words, is_split_into_words=True, truncation=False, add_special_tokens=True
        )
        lengths.append(len(enc["input_ids"]))
        encode_window(ex, tokenizer, max_length)  # raises on gold truncation

    ordered = sorted(lengths)
    return {
        "max_length_configured": max_length,
        "subword_len_max": ordered[-1],
        "subword_len_p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
        "subword_len_mean": sum(ordered) / len(ordered),
        "num_over_max_length": sum(1 for x in ordered if x > max_length),
        "gold_span_truncations": 0,
    }
