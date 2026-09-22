"""Breakdown and paired-statistics modules.

The breakdown code re-implements the official matching semantics for per-bucket
attribution, so it is pinned against the real evaluator here: on gold-as-
prediction every bucket must be perfect, and the totals must equal the official
ones.
"""

import pytest

from conftest import DATA_DIR, OFFICIAL_EVALUATOR

from scievent_tars.data.reader import read_split
from scievent_tars.evaluation.bootstrap import paired_bootstrap, per_window_counts
from scievent_tars.evaluation.breakdowns import (
    boundary_error_report,
    compute_breakdowns,
    exact_match,
    iou_match,
    role_support_split,
)
from scievent_tars.evaluation.export_official import export_predictions
from scievent_tars.evaluation.official_wrapper import score
from scievent_tars.modeling.model import PredictedEvent

GOLD = DATA_DIR / "dev.oneie.json"


def _gold_as_predictions():
    return {
        ex.wnd_id: [
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
        for ex in read_split(GOLD)
    }


@pytest.fixture(scope="module")
def perfect_predictions(tmp_path_factory):
    path = tmp_path_factory.mktemp("analysis") / "perfect.oneie.json"
    export_predictions(GOLD, _gold_as_predictions(), path)
    return path


def test_iou_and_exact_match_follow_the_upstream_contract():
    assert iou_match((0, 3), (0, 6)) is False  # exactly 0.5 -> no match
    assert iou_match((0, 4), (0, 7)) is True
    assert exact_match((2, 5), (2, 5)) is True
    assert exact_match((2, 5), (2, 6)) is False


def test_breakdowns_are_perfect_on_gold(perfect_predictions):
    breakdowns = compute_breakdowns(GOLD, perfect_predictions)
    assert set(breakdowns) >= {
        "domain", "event_type", "role", "arg_width", "window_length",
        "action_distance", "cross_sentence", "same_role_multiplicity",
    }
    for dimension, buckets in breakdowns.items():
        for key, stats in buckets.items():
            if not stats["gold_total"]:
                continue
            if stats.get("metric") == "recall_only":
                # Gold-defined bucket: only recall is meaningful.
                assert stats["recall"] == pytest.approx(100.0), f"{dimension}/{key}"
                assert "f1" not in stats and "precision" not in stats
            else:
                assert stats["f1"] == pytest.approx(100.0), f"{dimension}/{key}"


def test_gold_only_dimensions_do_not_report_a_fake_precision(perfect_predictions):
    breakdowns = compute_breakdowns(GOLD, perfect_predictions)
    for key, stats in breakdowns["same_role_multiplicity"].items():
        assert stats["metric"] == "recall_only"
        assert "precision" not in stats, (
            "a gold-defined bucket has no prediction side; a 0.0 precision would "
            "be an artifact"
        )


def test_breakdown_totals_match_the_official_evaluator(perfect_predictions):
    official = score(OFFICIAL_EVALUATOR, GOLD, perfect_predictions)["metrics"]
    breakdowns = compute_breakdowns(GOLD, perfect_predictions)
    gold_total = sum(v["gold_total"] for v in breakdowns["role"].values())
    counts = per_window_counts(GOLD, perfect_predictions)
    assert gold_total == sum(c["gold"] for c in counts.values())
    assert official["arg_c_iou_f1"] == pytest.approx(100.0)


def test_empty_predictions_give_zero_breakdowns(tmp_path):
    path = tmp_path / "empty.oneie.json"
    export_predictions(GOLD, {}, path)
    breakdowns = compute_breakdowns(GOLD, path)
    for buckets in breakdowns.values():
        for stats in buckets.values():
            assert stats["matched"] == 0
            assert stats["recall"] == 0.0
            if stats.get("metric") != "recall_only":
                assert stats["f1"] == 0.0


def test_boundary_error_report_has_no_gap_on_gold(perfect_predictions):
    report = boundary_error_report(GOLD, perfect_predictions)
    assert report["arg_i_exact"]["f1"] == pytest.approx(100.0)
    assert report["arg_i_iou"]["f1"] == pytest.approx(100.0)
    assert report["boundary_only_gap_f1"] == pytest.approx(0.0)


def test_role_support_split_flags_the_rare_roles():
    split = role_support_split(GOLD, threshold=100)
    assert "Ethical" in split["rare"]
    assert "Context" in split["frequent"]
    assert set(split["frequent"]) & set(split["rare"]) == set()


def test_paired_bootstrap_of_a_system_against_itself_is_zero(perfect_predictions):
    result = paired_bootstrap(
        GOLD, perfect_predictions, perfect_predictions, num_samples=200, seed=7
    )
    assert result["observed_delta_f1"] == pytest.approx(0.0)
    assert result["ci95_low"] == pytest.approx(0.0)
    assert result["ci95_high"] == pytest.approx(0.0)
    assert result["resampling_unit"] == "doc_id"
    assert result["num_document_blocks"] > 1


def test_paired_bootstrap_detects_a_large_real_difference(tmp_path, perfect_predictions):
    empty = tmp_path / "empty.oneie.json"
    export_predictions(GOLD, {}, empty)
    result = paired_bootstrap(GOLD, empty, perfect_predictions, num_samples=200, seed=7)
    assert result["f1_a"] == pytest.approx(0.0)
    assert result["f1_b"] == pytest.approx(100.0)
    assert result["observed_delta_f1"] == pytest.approx(100.0)
    assert result["bootstrap_p_value"] < 0.05


def test_bootstrap_rejects_mismatched_window_sets(tmp_path, perfect_predictions):
    lines = [l for l in open(perfect_predictions, encoding="utf-8") if l.strip()]
    short = tmp_path / "short.oneie.json"
    short.write_text("".join(lines[:-3]), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from gold"):
        paired_bootstrap(GOLD, short, perfect_predictions, num_samples=10)
