"""Every gold span must survive word -> subword pooling -> exported word coords.

Acceptance for Phase 7 step 11.1 and gate STOP_MODEL_01.
"""

import pytest

from conftest import BACKBONE, DATA_DIR

from scievent_tars.data.reader import read_split
from scievent_tars.data.schema import SEMANTIC_ROLES, TRIGGER_COMPONENTS
from scievent_tars.data.tokenizer_map import audit_lengths, encode_window, load_tokenizer, roundtrip_span

MAX_LENGTH = 768


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(str(BACKBONE))


@pytest.fixture(scope="module")
def splits():
    return {s: read_split(DATA_DIR / f"{s}.json") for s in ("train", "dev", "test")}


def test_all_gold_spans_roundtrip(tokenizer, splits):
    checked = 0
    for name, examples in splits.items():
        for ex in examples:
            enc = encode_window(ex, tokenizer, MAX_LENGTH)
            assert enc.num_words == ex.num_words, f"{name}/{ex.wnd_id}: word count changed"
            for event in ex.events:
                spans = [
                    event.action_span,
                    *event.agent_spans,
                    *event.primary_object_spans,
                    *event.secondary_object_spans,
                    *(a.span for a in event.arguments),
                ]
                for span in spans:
                    assert roundtrip_span(enc, span) == span, (
                        f"{name}/{ex.wnd_id}: {span.as_tuple()} did not round-trip"
                    )
                    checked += 1
    assert checked > 0


def test_no_example_is_silently_truncated(tokenizer, splits):
    for name, examples in splits.items():
        audit = audit_lengths(examples, tokenizer, MAX_LENGTH)
        assert audit["num_over_max_length"] == 0, (
            f"{name}: {audit['num_over_max_length']} windows exceed max_length="
            f"{MAX_LENGTH} (max observed {audit['subword_len_max']})"
        )


def test_word_piece_map_is_contiguous_and_ordered(tokenizer, splits):
    for examples in splits.values():
        for ex in examples[:200]:
            enc = encode_window(ex, tokenizer, MAX_LENGTH)
            previous_end = None
            for first, last in enc.word_piece_spans:
                assert last > first
                if previous_end is not None:
                    assert first >= previous_end
                previous_end = last


def test_trigger_components_separated_from_semantic_roles(splits):
    for name, examples in splits.items():
        for ex in examples:
            for event in ex.events:
                roles = {a.role for a in event.arguments}
                assert roles <= set(SEMANTIC_ROLES), f"{name}/{ex.wnd_id}: {roles}"
                assert not roles & set(TRIGGER_COMPONENTS)


def test_all_event_mentions_preserved(splits):
    import json

    for name, examples in splits.items():
        raw = [json.loads(l) for l in open(DATA_DIR / f"{name}.json", encoding="utf-8") if l.strip()]
        by_id = {r.get("wnd_id") or r["sent_id"]: r for r in raw}
        for ex in examples:
            record = by_id[ex.wnd_id]
            assert len(ex.events) == len(record["event_mentions"])
            gold_args = sum(len(e["arguments"]) for e in record["event_mentions"])
            parsed_args = sum(
                len(e.arguments)
                + len(e.agent_spans)
                + len(e.primary_object_spans)
                + len(e.secondary_object_spans)
                for e in ex.events
            )
            assert parsed_args == gold_args, f"{name}/{ex.wnd_id}: argument count changed"
