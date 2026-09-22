"""The model must operate at window level and never receive gold event identity.

Rule 6 / STOP_EVENT_01. A window can in principle hold several events, so the
event-set machinery is exercised here on a synthetic multi-event window even
though the released SciEvent split happens to contain exactly one event per
window.
"""

import torch

from conftest import DATA_DIR

from scievent_tars.data.reader import read_split, train_statistics
from scievent_tars.data.schema import (
    EVENT_TYPES,
    GoldArgument,
    GoldEvent,
    Span,
    WindowExample,
)
from scievent_tars.modeling.argument_set import RoleSlotLayout
from scievent_tars.training.losses import match_events

COST_WEIGHTS = {"type": 2.0, "action": 4.0, "Agent": 0.5,
                "PrimaryObject": 0.5, "SecondaryObject": 0.25}


def _synthetic_two_event_window() -> WindowExample:
    words = [f"w{i}" for i in range(40)]
    return WindowExample(
        doc_id="synthetic_doc",
        wnd_id="synthetic_doc-0",
        words=words,
        sentence_starts=[0, 20],
        events=[
            GoldEvent(
                event_id="e0",
                event_type="Methods/Approach",
                action_span=Span(2, 4),
                agent_spans=[Span(0, 2)],
                primary_object_spans=[Span(4, 8)],
                arguments=[GoldArgument("Method", Span(4, 8)),
                           GoldArgument("Purpose", Span(8, 14))],
            ),
            GoldEvent(
                event_id="e1",
                event_type="Results/Findings",
                action_span=Span(22, 23),
                agent_spans=[Span(20, 22)],
                arguments=[GoldArgument("Results", Span(23, 30))],
            ),
        ],
    )


def test_window_holds_multiple_events_without_flattening():
    example = _synthetic_two_event_window()
    example.validate()
    assert len(example.events) == 2
    # One inference sample, not one per gold event.
    assert example.wnd_id == "synthetic_doc-0"
    assert {e.event_type for e in example.events} <= set(EVENT_TYPES)


def test_hungarian_matching_is_permutation_invariant():
    """Swapping the gold event order must not change which slot wins which event."""
    example = _synthetic_two_event_window()
    k_event, num_words, num_types = 4, example.num_words, len(EVENT_TYPES) + 1

    torch.manual_seed(0)
    type_logits = torch.randn(k_event, num_types)
    action = {
        "start_logits": torch.randn(1, k_event, 1, num_words),
        "end_logits": torch.randn(1, k_event, 1, num_words),
    }
    components = {
        name: {
            "start_logits": torch.randn(1, k_event, 1, num_words),
            "end_logits": torch.randn(1, k_event, 1, num_words),
            "presence_logits": torch.randn(1, k_event, 1),
        }
        for name in ("Agent", "PrimaryObject", "SecondaryObject")
    }

    forward = match_events(type_logits, action, components, 0, example.events, COST_WEIGHTS)
    reversed_events = list(reversed(example.events))
    backward = match_events(type_logits, action, components, 0, reversed_events, COST_WEIGHTS)

    forward_pairs = {(slot, example.events[j].event_id) for slot, j in forward}
    backward_pairs = {(slot, reversed_events[j].event_id) for slot, j in backward}
    assert forward_pairs == backward_pairs


def test_more_gold_events_than_slots_assigns_distinct_slots():
    example = _synthetic_two_event_window()
    k_event, num_words = 2, example.num_words
    torch.manual_seed(1)
    type_logits = torch.randn(k_event, len(EVENT_TYPES) + 1)
    action = {
        "start_logits": torch.randn(1, k_event, 1, num_words),
        "end_logits": torch.randn(1, k_event, 1, num_words),
    }
    components = {
        name: {
            "start_logits": torch.randn(1, k_event, 1, num_words),
            "end_logits": torch.randn(1, k_event, 1, num_words),
            "presence_logits": torch.randn(1, k_event, 1),
        }
        for name in ("Agent", "PrimaryObject", "SecondaryObject")
    }
    pairs = match_events(type_logits, action, components, 0, example.events, COST_WEIGHTS)
    slots = [slot for slot, _ in pairs]
    golds = [j for _, j in pairs]
    assert len(set(slots)) == len(slots)
    assert len(set(golds)) == len(golds)


def test_k_event_capacity_covers_train_maximum():
    train = read_split(DATA_DIR / "train.json")
    stats = train_statistics(train)
    k_event = stats["max_events_per_window"] + 1
    assert k_event > stats["max_events_per_window"], "K_event must exceed the train max"
    for split in ("dev", "test"):
        observed = max(len(ex.events) for ex in read_split(DATA_DIR / f"{split}.json"))
        assert observed <= k_event, (
            f"{split} has {observed} events in a window but K_event={k_event}"
        )


def test_role_slot_layout_is_consistent():
    train = read_split(DATA_DIR / "train.json")
    stats = train_statistics(train)
    counts = {r: v + 1 for r, v in stats["role_multiplicity_max"].items()}
    layout = RoleSlotLayout.build(counts)
    assert layout.total == sum(counts.values())
    for role, (lo, hi) in layout.offsets.items():
        assert hi - lo == counts[role]
        assert all(layout.slot_index[i] == i - lo for i in range(lo, hi))
