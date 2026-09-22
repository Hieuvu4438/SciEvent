"""Export window-level predictions into the exact official OneIE gold schema.

One JSON object per **gold** window, always — a window with no predicted event
gets an empty ``event_mentions`` list rather than being omitted, so the
prediction ID set equals the gold ID set (``STOP_METRIC_02``).

Object keys mirror the generated gold files; no new evaluator input format is
invented.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.schema import EVENT_TYPES, SEMANTIC_ROLES, TRIGGER_COMPONENTS, Span
from ..modeling.model import PredictedEvent

_VALID_ROLES = set(SEMANTIC_ROLES) | set(TRIGGER_COMPONENTS)


def _mention_text(tokens: list[str], span: Span) -> str:
    return " ".join(tokens[span.start : span.end])


def build_prediction_record(
    gold_record: dict, events: list[PredictedEvent]
) -> dict:
    sent_id = gold_record.get("sent_id") or gold_record["wnd_id"]
    tokens = gold_record["tokens"]
    num_words = len(tokens)

    entity_mentions: list[dict] = []
    entity_ids: dict[tuple[int, int], str] = {}

    def entity_id_for(span: Span) -> str:
        key = span.as_tuple()
        if key not in entity_ids:
            eid = f"{sent_id}-E{len(entity_mentions)}"
            entity_ids[key] = eid
            entity_mentions.append(
                {
                    "id": eid,
                    "text": _mention_text(tokens, span),
                    "start": span.start,
                    "end": span.end,
                    "entity_type": "UNK",
                    "mention_type": "UNK",
                }
            )
        return entity_ids[key]

    event_mentions: list[dict] = []
    for k, ev in enumerate(events):
        if ev.event_type not in EVENT_TYPES:
            raise ValueError(f"{sent_id}: predicted event type outside schema: {ev.event_type!r}")
        if not (0 <= ev.action_span.start < ev.action_span.end <= num_words):
            raise ValueError(f"{sent_id}: invalid trigger span {ev.action_span.as_tuple()}")

        arguments: list[dict] = []
        for comp in TRIGGER_COMPONENTS:
            for span in ev.components.get(comp, []):
                if not (0 <= span.start < span.end <= num_words):
                    raise ValueError(f"{sent_id}: invalid {comp} span {span.as_tuple()}")
                arguments.append(
                    {
                        "entity_id": entity_id_for(span),
                        "text": _mention_text(tokens, span),
                        "role": comp,
                    }
                )
        for role, span in ev.arguments:
            if role not in _VALID_ROLES:
                raise ValueError(f"{sent_id}: role outside official schema: {role!r}")
            if not (0 <= span.start < span.end <= num_words):
                raise ValueError(f"{sent_id}: invalid argument span {span.as_tuple()}")
            arguments.append(
                {
                    "entity_id": entity_id_for(span),
                    "text": _mention_text(tokens, span),
                    "role": role,
                }
            )

        event_mentions.append(
            {
                "event_type": ev.event_type,
                "id": f"{sent_id}-EV{k}",
                "trigger": {
                    "text": _mention_text(tokens, ev.action_span),
                    "start": ev.action_span.start,
                    "end": ev.action_span.end,
                },
                "arguments": arguments,
            }
        )

    return {
        "doc_id": gold_record["doc_id"],
        "sent_id": sent_id,
        "entity_mentions": entity_mentions,
        "relation_mentions": [],
        "event_mentions": event_mentions,
        "entity_coreference": [],
        "event_coreference": [],
        "tokens": tokens,
        "pieces": gold_record.get("pieces", []),
        "token_lens": gold_record.get("token_lens", []),
        "sentence": gold_record.get("sentence", " ".join(tokens)),
        "sentence_starts": gold_record.get("sentence_starts", [0]),
    }


def export_predictions(
    gold_path: str | Path,
    predictions: dict[str, list[PredictedEvent]],
    out_path: str | Path,
) -> dict:
    """Write an official-format JSONL aligned one-to-one with the gold file."""
    with open(gold_path, "r", encoding="utf-8") as fh:
        gold_records = [json.loads(line) for line in fh if line.strip()]

    gold_ids = {r.get("sent_id") or r["wnd_id"] for r in gold_records}
    unknown = sorted(set(predictions) - gold_ids)
    if unknown:
        raise ValueError(f"predictions contain unknown window ids: {unknown[:5]}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_events = 0
    num_arguments = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in gold_records:
            sent_id = record.get("sent_id") or record["wnd_id"]
            events = predictions.get(sent_id, [])
            num_events += len(events)
            num_arguments += sum(len(e.arguments) for e in events)
            fh.write(json.dumps(build_prediction_record(record, events)) + "\n")

    return {
        "num_windows": len(gold_records),
        "num_predicted_events": num_events,
        "num_predicted_semantic_arguments": num_arguments,
        "path": str(out_path),
    }
