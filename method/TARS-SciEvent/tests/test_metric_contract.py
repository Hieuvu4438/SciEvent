"""
tests/test_metric_contract.py

Synthetic unit tests verifying that our understanding of the official evaluator
(vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py) matches upstream behavior exactly.
"""

import sys

# Add official evaluator directory to path
EVAL_DIR = "/home/haipd/SciEvent/third_party/SciEvent/baselines/ONEIE"
sys.path.insert(0, EVAL_DIR)

from EM_overlap_eval import iou_overlap, compute_f1


def test_iou_strict_inequality():
    """Verify IoU requires intersection / union > 0.5, NOT >= 0.5."""
    # [0, 2] and [0, 4]: inter = 2, union = 4 -> IoU = 2/4 = 0.5
    # Official contract: 2/4 > 0.5 is False
    assert iou_overlap((0, 2), (0, 4)) is False

    # [0, 3] and [0, 5]: inter = 3, union = 5 -> IoU = 3/5 = 0.6 > 0.5 -> True
    assert iou_overlap((0, 3), (0, 5)) is True

    # Identical spans: [0, 4] and [0, 4] -> IoU = 1.0 > 0.5 -> True
    assert iou_overlap((0, 4), (0, 4)) is True


def test_roles_exclusion():
    """Verify Agent, PrimaryObject, SecondaryObject are excluded from semantic arguments."""
    pred = {
        "sent_1": [
            ((0, 1, "Methods/Approach"), (2, 5, "Agent")),
            ((0, 1, "Methods/Approach"), (5, 8, "PrimaryObject")),
            ((0, 1, "Methods/Approach"), (8, 10, "Method")),
        ]
    }
    gold = {
        "sent_1": [
            ((0, 1, "Methods/Approach"), (2, 5, "Agent")),
            ((0, 1, "Methods/Approach"), (5, 8, "PrimaryObject")),
            ((0, 1, "Methods/Approach"), (8, 10, "Method")),
        ]
    }

    def match_arg_c(p, g):
        return p[1][2] == g[1][2] and iou_overlap((p[1][0], p[1][1]), (g[1][0], g[1][1]))

    prec, rec, f1, matched, total_pred, total_gold = compute_f1(pred, gold, match_arg_c)
    # Only 1 semantic role ("Method") should be counted
    assert matched == 1
    assert total_pred == 1
    assert total_gold == 1
    assert f1 == 1.0
    assert prec == 1.0
    assert rec == 1.0


def test_one_to_one_greedy_matching():
    """Verify duplicate predictions against one gold yield only one TP and extra FPs."""
    pred = {
        "sent_1": [
            ((0, 1, "Methods/Approach"), (8, 10, "Method")),
            ((0, 1, "Methods/Approach"), (8, 10, "Method")),  # duplicate
        ]
    }
    gold = {
        "sent_1": [
            ((0, 1, "Methods/Approach"), (8, 10, "Method")),
        ]
    }

    def match_arg_c(p, g):
        return p[1][2] == g[1][2] and iou_overlap((p[1][0], p[1][1]), (g[1][0], g[1][1]))

    prec, rec, f1, matched, total_pred, total_gold = compute_f1(pred, gold, match_arg_c)
    # Matched = 1, total_pred = 2, total_gold = 1
    # Prec = 1/2 = 0.5, Rec = 1/1 = 1.0, F1 = 2*(0.5*1.0)/(1.5) = 2/3 = 0.6667
    assert matched == 1
    assert total_pred == 2
    assert total_gold == 1
    assert prec == 0.5
    assert rec == 1.0
    assert round(f1, 4) == 0.6667


def test_empty_predictions_and_gold():
    """Verify empty pred and empty gold handle cleanly without crashing."""
    def match_arg_c(p, g):
        return True

    # Empty pred
    prec, rec, f1, matched, total_pred, total_gold = compute_f1({}, {"sent_1": [((0, 1, "Method"), (2, 4, "Purpose"))]}, match_arg_c)
    assert prec == 0 and rec == 0 and f1 == 0 and matched == 0

    # Empty gold
    prec, rec, f1, matched, total_pred, total_gold = compute_f1({"sent_1": [((0, 1, "Method"), (2, 4, "Purpose"))]}, {}, match_arg_c)
    assert prec == 0 and rec == 0 and f1 == 0 and matched == 0


def _arg_c(overlap_fn):
    def match(p, g):
        return p[1][2] == g[1][2] and overlap_fn((p[1][0], p[1][1]), (g[1][0], g[1][1]))
    return match


def _arg_i(overlap_fn):
    def match(p, g):
        return overlap_fn((p[1][0], p[1][1]), (g[1][0], g[1][1]))
    return match


def test_iou_just_above_half_matches():
    """IoU 0.5 + epsilon must match; the boundary itself must not."""
    # inter = 3, union = 6 -> exactly 0.5 -> no match
    assert iou_overlap((0, 3), (0, 6)) is False
    # inter = 4, union = 7 -> 0.5714 -> match
    assert iou_overlap((0, 4), (0, 7)) is True
    # disjoint, touching at a point -> no match
    assert iou_overlap((0, 5), (5, 10)) is False


def test_one_prediction_against_duplicate_gold_yields_one_tp():
    etype = "Results/Findings"
    pred = {"s": [((0, 1, etype), (4, 8, "Results"))]}
    gold = {"s": [((0, 1, etype), (4, 8, "Results")),
                  ((0, 1, etype), (4, 8, "Results"))]}
    _, _, _, matched, total_pred, total_gold = compute_f1(pred, gold, _arg_c(iou_overlap))
    assert (matched, total_pred, total_gold) == (1, 1, 2)


def test_correct_span_wrong_role_is_arg_i_hit_arg_c_miss():
    etype = "Background/Introduction"
    pred = {"s": [((0, 1, etype), (3, 9, "Purpose"))]}
    gold = {"s": [((0, 1, etype), (3, 9, "Context"))]}

    _, _, _, matched_i, _, _ = compute_f1(pred, gold, _arg_i(iou_overlap))
    _, _, _, matched_c, _, _ = compute_f1(pred, gold, _arg_c(iou_overlap))
    assert matched_i == 1
    assert matched_c == 0


def test_partial_span_is_iou_hit_but_exact_miss():
    etype = "Methods/Approach"
    # inter = 7, union = 10 -> IoU 0.7 -> IoU match, exact mismatch
    pred = {"s": [((0, 1, etype), (0, 7, "Method"))]}
    gold = {"s": [((0, 1, etype), (0, 10, "Method"))]}

    _, _, _, matched_iou, _, _ = compute_f1(pred, gold, _arg_c(iou_overlap))
    _, _, _, matched_exact, _, _ = compute_f1(
        pred, gold, lambda p, g: p[1] == g[1]
    )
    assert matched_iou == 1
    assert matched_exact == 0


def test_event_type_must_match_for_any_credit():
    """compute_f1 short-circuits on p[0][2] != g[0][2]: a wrong type scores zero."""
    pred = {"s": [((0, 1, "Methods/Approach"), (3, 9, "Method"))]}
    gold = {"s": [((0, 1, "Results/Findings"), (3, 9, "Method"))]}
    _, _, _, matched, total_pred, total_gold = compute_f1(pred, gold, _arg_c(iou_overlap))
    assert matched == 0
    assert (total_pred, total_gold) == (1, 1)


def test_trigger_span_is_ignored_for_argument_scoring():
    etype = "Conclusions/Implications"
    pred = {"s": [((90, 95, etype), (3, 9, "Implications"))]}
    gold = {"s": [((0, 1, etype), (3, 9, "Implications"))]}
    _, _, _, matched, _, _ = compute_f1(pred, gold, _arg_c(iou_overlap))
    assert matched == 1, "the official argument score must be trigger-insensitive"


def test_unknown_pred_sent_id_is_ignored_upstream():
    """Documented upstream behaviour: compute_f1 iterates gold ids only.

    Predictions under an id absent from gold are silently dropped -- which is
    exactly why our wrapper asserts set equality before scoring (STOP_METRIC_02).
    """
    etype = "Methods/Approach"
    pred = {
        "known": [((0, 1, etype), (3, 9, "Method"))],
        "ghost": [((0, 1, etype), (3, 9, "Method"))] * 50,
    }
    gold = {"known": [((0, 1, etype), (3, 9, "Method"))]}
    prec, rec, f1, matched, total_pred, total_gold = compute_f1(
        pred, gold, _arg_c(iou_overlap)
    )
    assert (matched, total_pred, total_gold) == (1, 1, 1)
    assert f1 == 1.0  # the 50 ghost false positives never cost anything


def test_missing_pred_sent_id_counts_as_pure_recall_loss():
    etype = "Methods/Approach"
    gold = {
        "a": [((0, 1, etype), (3, 9, "Method"))],
        "b": [((0, 1, etype), (3, 9, "Method"))],
    }
    pred = {"a": [((0, 1, etype), (3, 9, "Method"))]}
    prec, rec, _, matched, total_pred, total_gold = compute_f1(
        pred, gold, _arg_c(iou_overlap)
    )
    assert (matched, total_pred, total_gold) == (1, 1, 2)
    assert prec == 1.0 and rec == 0.5
