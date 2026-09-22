"""Canonical TARS-SciEvent data model.

All spans are **word-token, end-exclusive** coordinates, exactly matching the
official SciEvent / OneIE representation. Subword offsets never leave the
tokenizer-mapping layer.

The label inventories below are the ones that actually occur in the official
released annotations (see ``data/manifests/split_audit.json``). The playbook
uses shorthand names (``Background``, ``Result``, ...); the released files use
the full names reproduced here, and the official evaluator compares raw
strings, so these are authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --- Official label inventories -------------------------------------------------

EVENT_TYPES: tuple[str, ...] = (
    "Background/Introduction",
    "Methods/Approach",
    "Results/Findings",
    "Conclusions/Implications",
)

#: The nine semantic roles scored by the official evaluator.
SEMANTIC_ROLES: tuple[str, ...] = (
    "Context",
    "Purpose",
    "Method",
    "Results",
    "Analysis",
    "Challenge",
    "Ethical",
    "Implications",
    "Contradictions",
)

#: Trigger-structure components. The official evaluator explicitly excludes
#: these from Arg-I / Arg-C (``exclude_roles`` in ``EM_overlap_eval.py``).
TRIGGER_COMPONENTS: tuple[str, ...] = ("Agent", "PrimaryObject", "SecondaryObject")

ALL_ROLES: tuple[str, ...] = SEMANTIC_ROLES + TRIGGER_COMPONENTS

EVENT_TYPE_TO_ID = {t: i for i, t in enumerate(EVENT_TYPES)}
#: Index of the "no event in this slot" class, appended after the real types.
NONE_EVENT_ID = len(EVENT_TYPES)

SEMANTIC_ROLE_TO_ID = {r: i for i, r in enumerate(SEMANTIC_ROLES)}
TRIGGER_COMPONENT_TO_ID = {c: i for i, c in enumerate(TRIGGER_COMPONENTS)}


# --- Core objects ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open word span ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start < 0: {self}")
        if self.end <= self.start:
            raise ValueError(f"span end <= start: {self}")

    @property
    def width(self) -> int:
        return self.end - self.start

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)

    def iou(self, other: "Span") -> float:
        """Official IoU as implemented upstream (``iou_overlap``)."""
        inter = min(self.end, other.end) - max(self.start, other.start)
        if inter <= 0:
            return 0.0
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union

    def matches_official(self, other: "Span") -> bool:
        """Upstream contract: ``intersection / union > 0.5`` (strict)."""
        return self.iou(other) > 0.5


@dataclass(frozen=True, slots=True)
class GoldArgument:
    """One of the nine *semantic* roles."""

    role: str
    span: Span

    def __post_init__(self) -> None:
        if self.role not in SEMANTIC_ROLE_TO_ID:
            raise ValueError(f"unknown semantic role: {self.role!r}")


@dataclass(slots=True)
class GoldEvent:
    event_id: str
    event_type: str
    action_span: Span
    agent_spans: list[Span] = field(default_factory=list)
    primary_object_spans: list[Span] = field(default_factory=list)
    secondary_object_spans: list[Span] = field(default_factory=list)
    arguments: list[GoldArgument] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPE_TO_ID:
            raise ValueError(f"unknown event type: {self.event_type!r}")

    def component_spans(self, component: str) -> list[Span]:
        return {
            "Agent": self.agent_spans,
            "PrimaryObject": self.primary_object_spans,
            "SecondaryObject": self.secondary_object_spans,
        }[component]

    def spans_for_role(self, role: str) -> list[Span]:
        return [a.span for a in self.arguments if a.role == role]


@dataclass(slots=True)
class WindowExample:
    """One SciEvent window. The unit of inference — never one sample per event."""

    doc_id: str
    wnd_id: str
    words: list[str]
    sentence_starts: list[int]
    events: list[GoldEvent] = field(default_factory=list)
    domain: str | None = None

    @property
    def num_words(self) -> int:
        return len(self.words)

    def sentence_id_of_word(self) -> list[int]:
        """Map each word index to its 0-based sentence index."""
        starts = sorted(s for s in self.sentence_starts if 0 <= s < self.num_words)
        if not starts or starts[0] != 0:
            starts = [0] + starts
        out: list[int] = []
        sent = -1
        nxt = 0
        for i in range(self.num_words):
            while nxt < len(starts) and starts[nxt] == i:
                sent += 1
                nxt += 1
            out.append(max(sent, 0))
        return out

    def all_semantic_spans(self) -> list[Span]:
        return [a.span for ev in self.events for a in ev.arguments]

    def validate(self) -> None:
        n = self.num_words
        if n == 0:
            raise ValueError(f"{self.wnd_id}: empty window")
        for ev in self.events:
            for label, spans in (
                ("action", [ev.action_span]),
                ("Agent", ev.agent_spans),
                ("PrimaryObject", ev.primary_object_spans),
                ("SecondaryObject", ev.secondary_object_spans),
                ("argument", [a.span for a in ev.arguments]),
            ):
                for sp in spans:
                    if sp.end > n:
                        raise ValueError(
                            f"{self.wnd_id}: {label} span {sp.as_tuple()} exceeds "
                            f"{n} words"
                        )


def infer_domain(doc_id: str) -> str:
    """Domain key exactly as the official evaluator derives it."""
    return doc_id.strip().split("_")[0]


def check_role_inventory(roles: Iterable[str]) -> None:
    unknown = sorted(set(roles) - set(ALL_ROLES))
    if unknown:
        raise ValueError(f"roles outside the official schema: {unknown}")


def check_event_types(types: Sequence[str]) -> None:
    unknown = sorted(set(types) - set(EVENT_TYPES))
    if unknown:
        raise ValueError(f"event types outside the official schema: {unknown}")
