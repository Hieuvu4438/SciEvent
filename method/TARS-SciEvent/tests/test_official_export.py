"""Export + strict-wrapper contract against the untouched official evaluator.

The decisive test is the identity round-trip: exporting the gold structure as
if it were our prediction must score 100 through the real upstream evaluator.
If that fails, our exporter and the evaluator disagree about the schema.
"""

import json

import pytest

from conftest import DATA_DIR, OFFICIAL_EVALUATOR

from scievent_tars.data.reader import read_split
from scievent_tars.evaluation.export_official import export_predictions
from scievent_tars.evaluation.official_wrapper import (
    PredictionContractError,
    score,
    validate_predictions,
)
from scievent_tars.modeling.model import PredictedEvent
from scievent_tars.data.schema import Span

GOLD = DATA_DIR / "dev.oneie.json"


def _gold_as_predictions() -> dict[str, list[PredictedEvent]]:
    predictions = {}
    for ex in read_split(GOLD):
        predictions[ex.wnd_id] = [
            PredictedEvent(
                event_type=ev.event_type,
                action_span=ev.action_span,
                components={
                    "Agent": list(ev.agent_spans),
                    "PrimaryObject": list(ev.primary_object_spans),
                    "SecondaryObject": list(ev.secondary_object_spans),
                },
                arguments=[(a.role, a.span) for a in ev.arguments],
            )
            for ev in ex.events
        ]
    return predictions


def test_gold_roundtrip_scores_perfect(tmp_path):
    pred_path = tmp_path / "dev.oneie.json"
    info = export_predictions(GOLD, _gold_as_predictions(), pred_path)
    assert info["num_windows"] == sum(1 for _ in open(GOLD, encoding="utf-8"))

    payload = score(OFFICIAL_EVALUATOR, GOLD, pred_path, require_exact_id_set=True)
    metrics = payload["metrics"]
    for key in ("arg_c_exact_f1", "arg_c_iou_f1", "arg_i_exact_f1", "arg_i_iou_f1"):
        assert metrics[key] == pytest.approx(100.0, abs=1e-6), f"{key}={metrics[key]}"


def test_empty_predictions_export_every_window(tmp_path):
    pred_path = tmp_path / "empty.oneie.json"
    info = export_predictions(GOLD, {}, pred_path)
    assert info["num_predicted_events"] == 0

    records = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    assert all(r["event_mentions"] == [] for r in records)
    # A window with no predicted event is exported, never omitted.
    assert validate_predictions(pred_path, GOLD)["id_sets_equal"]

    payload = score(OFFICIAL_EVALUATOR, GOLD, pred_path)
    assert payload["metrics"]["arg_c_iou_f1"] == 0.0


def test_missing_window_is_rejected(tmp_path):
    lines = [l for l in open(GOLD, encoding="utf-8") if l.strip()]
    truncated = tmp_path / "missing.oneie.json"
    truncated.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(PredictionContractError, match="STOP_METRIC_02"):
        validate_predictions(truncated, GOLD)


def test_unknown_window_is_rejected(tmp_path):
    lines = [l for l in open(GOLD, encoding="utf-8") if l.strip()]
    extra = json.loads(lines[0])
    extra["sent_id"] = "not_a_real_window-99"
    path = tmp_path / "extra.oneie.json"
    body = "".join(l if l.endswith("\n") else l + "\n" for l in lines)
    path.write_text(body + json.dumps(extra) + "\n", encoding="utf-8")
    with pytest.raises(PredictionContractError, match="STOP_METRIC_02"):
        validate_predictions(path, GOLD)


def test_out_of_range_span_is_rejected(tmp_path):
    lines = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    lines[0]["entity_mentions"][0]["end"] = len(lines[0]["tokens"]) + 5
    path = tmp_path / "bad_span.oneie.json"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    with pytest.raises(PredictionContractError, match="outside"):
        validate_predictions(path, GOLD)


def test_invalid_event_type_is_rejected(tmp_path):
    lines = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    for record in lines:
        if record["event_mentions"]:
            record["event_mentions"][0]["event_type"] = "NotAType"
            break
    path = tmp_path / "bad_type.oneie.json"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    with pytest.raises(PredictionContractError, match="event type outside schema"):
        validate_predictions(path, GOLD)


def test_exporter_rejects_span_beyond_window():
    gold_record = json.loads(open(GOLD, encoding="utf-8").readline())
    n = len(gold_record["tokens"])
    from scievent_tars.evaluation.export_official import build_prediction_record

    bad = [PredictedEvent(
        event_type="Methods/Approach",
        action_span=Span(0, 1),
        components={},
        arguments=[("Method", Span(n - 1, n + 3))],
    )]
    with pytest.raises(ValueError, match="invalid argument span"):
        build_prediction_record(gold_record, bad)


def test_trigger_components_do_not_affect_semantic_scores(tmp_path):
    """Arg-I/Arg-C must ignore Agent/PrimaryObject/SecondaryObject entirely."""
    with_components = _gold_as_predictions()
    without = {
        wid: [
            PredictedEvent(
                event_type=ev.event_type,
                action_span=ev.action_span,
                components={},
                arguments=ev.arguments,
            )
            for ev in events
        ]
        for wid, events in with_components.items()
    }
    a = tmp_path / "with.json"
    b = tmp_path / "without.json"
    export_predictions(GOLD, with_components, a)
    export_predictions(GOLD, without, b)

    ma = score(OFFICIAL_EVALUATOR, GOLD, a)["metrics"]
    mb = score(OFFICIAL_EVALUATOR, GOLD, b)["metrics"]
    for key in ("arg_c_exact_f1", "arg_c_iou_f1", "arg_i_iou_f1"):
        assert ma[key] == pytest.approx(mb[key])


def test_wrong_trigger_span_does_not_change_argument_scores(tmp_path):
    """The official argument score is trigger-span insensitive."""
    shifted = {
        wid: [
            PredictedEvent(
                event_type=ev.event_type,
                action_span=Span(0, 1),  # deliberately wrong trigger
                components=ev.components,
                arguments=ev.arguments,
            )
            for ev in events
        ]
        for wid, events in _gold_as_predictions().items()
    }
    path = tmp_path / "shifted.json"
    export_predictions(GOLD, shifted, path)
    metrics = score(OFFICIAL_EVALUATOR, GOLD, path)["metrics"]
    assert metrics["arg_c_iou_f1"] == pytest.approx(100.0, abs=1e-6)


def test_wrong_event_type_destroys_argument_scores(tmp_path):
    """The official argument score IS event-type sensitive."""
    flipped = {
        wid: [
            PredictedEvent(
                event_type=(
                    "Results/Findings"
                    if ev.event_type != "Results/Findings"
                    else "Methods/Approach"
                ),
                action_span=ev.action_span,
                components=ev.components,
                arguments=ev.arguments,
            )
            for ev in events
        ]
        for wid, events in _gold_as_predictions().items()
    }
    path = tmp_path / "flipped.json"
    export_predictions(GOLD, flipped, path)
    metrics = score(OFFICIAL_EVALUATOR, GOLD, path)["metrics"]
    assert metrics["arg_c_iou_f1"] == 0.0
