"""Read the frozen official SciEvent splits into :class:`WindowExample` objects.

Only ``data/official/`` is read. The vendor tree is never touched at train time
so an accidental re-preprocessing cannot mutate a running experiment.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .schema import (
    EVENT_TYPES,
    SEMANTIC_ROLE_TO_ID,
    TRIGGER_COMPONENTS,
    GoldArgument,
    GoldEvent,
    Span,
    WindowExample,
    check_event_types,
    check_role_inventory,
    infer_domain,
)


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _window_id(raw: dict) -> str:
    """Official files use ``wnd_id`` (DEGREE) or ``sent_id`` (OneIE)."""
    for key in ("wnd_id", "sent_id"):
        if key in raw:
            return raw[key]
    raise KeyError("record has neither wnd_id nor sent_id")


def parse_window(raw: dict) -> WindowExample:
    wnd_id = _window_id(raw)
    words = list(raw["tokens"])
    n = len(words)
    if n == 0:
        raise ValueError(f"{wnd_id}: reconstructed window has zero tokens")

    entity_map: dict[str, Span] = {}
    for ent in raw.get("entity_mentions", []):
        start, end = int(ent["start"]), int(ent["end"])
        if start < 0 or end <= start or end > n:
            raise ValueError(
                f"{wnd_id}: entity {ent['id']} has invalid span ({start}, {end}) "
                f"for {n} words"
            )
        entity_map[ent["id"]] = Span(start, end)

    events: list[GoldEvent] = []
    for ev in raw.get("event_mentions", []):
        trig = ev["trigger"]
        ts, te = int(trig["start"]), int(trig["end"])
        if ts < 0 or te <= ts or te > n:
            raise ValueError(f"{wnd_id}: invalid trigger span ({ts}, {te})")

        components: dict[str, list[Span]] = {c: [] for c in TRIGGER_COMPONENTS}
        arguments: list[GoldArgument] = []
        for arg in ev.get("arguments", []):
            role = arg["role"]
            eid = arg.get("entity_id")
            if eid is not None and eid in entity_map:
                span = entity_map[eid]
            elif "start" in arg and "end" in arg:
                span = Span(int(arg["start"]), int(arg["end"]))
            else:
                raise ValueError(f"{wnd_id}: argument without a resolvable span: {arg}")
            if span.end > n:
                raise ValueError(f"{wnd_id}: argument span {span.as_tuple()} out of range")
            if role in components:
                components[role].append(span)
            elif role in SEMANTIC_ROLE_TO_ID:
                arguments.append(GoldArgument(role=role, span=span))
            else:
                raise ValueError(f"{wnd_id}: role outside official schema: {role!r}")

        events.append(
            GoldEvent(
                event_id=ev["id"],
                event_type=ev["event_type"],
                action_span=Span(ts, te),
                agent_spans=components["Agent"],
                primary_object_spans=components["PrimaryObject"],
                secondary_object_spans=components["SecondaryObject"],
                arguments=arguments,
            )
        )

    doc_id = raw["doc_id"]
    example = WindowExample(
        doc_id=doc_id,
        wnd_id=wnd_id,
        words=words,
        sentence_starts=list(raw.get("sentence_starts") or [0]),
        events=events,
        domain=infer_domain(doc_id),
    )
    example.validate()
    return example


def read_split(path: str | Path) -> list[WindowExample]:
    """Parse one official split file, failing loudly on any schema violation."""
    raw_records = _read_jsonl(path)
    if not raw_records:
        raise ValueError(f"{path}: split is empty")

    examples = [parse_window(r) for r in raw_records]

    seen: set[str] = set()
    for ex in examples:
        if ex.wnd_id in seen:
            raise ValueError(f"{path}: duplicate window id {ex.wnd_id}")
        seen.add(ex.wnd_id)

    check_event_types([ev.event_type for ex in examples for ev in ex.events])
    check_role_inventory(
        [a.role for ex in examples for ev in ex.events for a in ev.arguments]
    )
    return examples


def read_official_splits(
    data_dir: str | Path, splits: tuple[str, ...] = ("train", "dev", "test")
) -> dict[str, list[WindowExample]]:
    data_dir = Path(data_dir)
    return {s: read_split(data_dir / f"{s}.json") for s in splits}


# --- TRAIN-only capacity statistics ---------------------------------------------


def train_statistics(examples: list[WindowExample]) -> dict:
    """Summarise TRAIN so architectural capacities are never taken from dev/test."""
    events_per_window = [len(ex.events) for ex in examples]
    widths = sorted(
        a.span.width for ex in examples for ev in ex.events for a in ev.arguments
    )
    role_counts: Counter[str] = Counter()
    role_multiplicity: dict[str, list[int]] = {r: [] for r in SEMANTIC_ROLE_TO_ID}
    component_multiplicity: dict[str, list[int]] = {c: [] for c in TRIGGER_COMPONENTS}
    type_counts: Counter[str] = Counter()

    for ex in examples:
        for ev in ex.events:
            type_counts[ev.event_type] += 1
            per_role: Counter[str] = Counter(a.role for a in ev.arguments)
            for role in role_multiplicity:
                role_multiplicity[role].append(per_role.get(role, 0))
                role_counts[role] += per_role.get(role, 0)
            for comp in component_multiplicity:
                component_multiplicity[comp].append(len(ev.component_spans(comp)))

    def percentile(values: list[int], q: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "num_windows": len(examples),
        "max_events_per_window": max(events_per_window) if events_per_window else 0,
        "event_type_counts": {t: type_counts.get(t, 0) for t in EVENT_TYPES},
        "semantic_role_counts": dict(role_counts),
        "semantic_arg_width": {
            "max": widths[-1] if widths else 0,
            "p99_5": percentile(widths, 0.995),
            "p99": percentile(widths, 0.99),
            "p95": percentile(widths, 0.95),
        },
        "role_multiplicity_max": {r: max(v) if v else 0 for r, v in role_multiplicity.items()},
        "role_multiplicity_p99": {r: percentile(v, 0.99) for r, v in role_multiplicity.items()},
        "component_multiplicity_max": {
            c: max(v) if v else 0 for c, v in component_multiplicity.items()
        },
    }
