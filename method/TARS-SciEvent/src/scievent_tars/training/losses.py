"""Hungarian set-prediction losses.

Three independent assignments:

* events        -- predicted event slots <-> gold events (permutation invariant)
* components    -- Action/Agent/PrimaryObject/SecondaryObject slots <-> gold spans
* arguments     -- role slots <-> the gold span set of that role, per role

Unmatched slots are trained toward NONE / NULL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..data.schema import (
    NONE_EVENT_ID,
    SEMANTIC_ROLES,
    TRIGGER_COMPONENTS,
    EVENT_TYPE_TO_ID,
    GoldEvent,
    Span,
)
from ..modeling.argument_set import RoleSlotLayout


@dataclass
class LossWeights:
    action: float = 1.0
    tuple_components: float = 1.0
    argument: float = 2.0
    boundary: float = 0.5
    prototype: float = 0.05
    event_type: float = 1.0
    #: DETR-style down-weighting of the (heavily dominant) NULL role slots.
    null_weight: float = 0.1
    boundary_pos_weight: float = 5.0


@dataclass
class LossBreakdown:
    total: torch.Tensor
    parts: dict[str, torch.Tensor] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)


def _log_softmax_last(x: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(x.float(), dim=-1)


def _span_neg_logprob(
    start_logp: torch.Tensor,  # [S, W]
    end_logp: torch.Tensor,  # [S, W]
    span: Span,
) -> torch.Tensor:
    return -start_logp[:, span.start] - end_logp[:, span.end - 1]


def _assign(cost: torch.Tensor) -> tuple[list[int], list[int]]:
    rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return rows.tolist(), cols.tolist()


def component_loss_for_slot(
    head_out: dict[str, torch.Tensor],
    batch_idx: int,
    event_slot: int,
    gold_spans: list[Span],
    nullable: bool,
) -> tuple[torch.Tensor, int]:
    """Loss for one component of one matched event slot; also returns dropped gold."""
    start_logp = _log_softmax_last(head_out["start_logits"][batch_idx, event_slot])  # [S, W]
    end_logp = _log_softmax_last(head_out["end_logits"][batch_idx, event_slot])
    n_slots = start_logp.size(0)
    presence = (
        head_out["presence_logits"][batch_idx, event_slot].float() if nullable else None
    )

    if not gold_spans:
        if presence is None:
            return start_logp.new_zeros(()), 0
        return -F.logsigmoid(-presence).sum(), 0

    usable = gold_spans[:n_slots]
    dropped = len(gold_spans) - len(usable)

    cost = torch.stack([_span_neg_logprob(start_logp, end_logp, g) for g in usable], dim=1)
    if presence is not None:
        cost = cost - F.logsigmoid(presence).unsqueeze(1)
    rows, cols = _assign(cost)

    loss = cost[rows, cols].sum()
    if presence is not None:
        unmatched = sorted(set(range(n_slots)) - set(rows))
        if unmatched:
            loss = loss - F.logsigmoid(-presence[unmatched]).sum()
    return loss, dropped


def argument_loss_for_event(
    arg_logits: torch.Tensor,  # [M, C + 1] for one (batch, event slot)
    layout: RoleSlotLayout,
    gold_event: GoldEvent,
    candidate_index: dict[tuple[int, int], int],
    null_index: int,
    null_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    logp = _log_softmax_last(arg_logits)  # [M, C+1]
    total = logp.new_zeros(())
    stats = {"gold_args": 0.0, "dropped_capacity": 0.0, "missing_candidate": 0.0}

    for role in SEMANTIC_ROLES:
        lo, hi = layout.offsets[role]
        slot_logp = logp[lo:hi]  # [K_r, C+1]
        n_slots = hi - lo

        spans = gold_event.spans_for_role(role)
        indices: list[int] = []
        for sp in spans:
            idx = candidate_index.get(sp.as_tuple())
            if idx is None:
                stats["missing_candidate"] += 1.0
                continue
            indices.append(idx)
        stats["gold_args"] += float(len(spans))

        if not indices:
            total = total - null_weight * slot_logp[:, null_index].sum()
            continue

        usable = indices[:n_slots]
        stats["dropped_capacity"] += float(len(indices) - len(usable))

        cost = -slot_logp[:, usable]  # [K_r, G]
        rows, cols = _assign(cost)
        total = total + cost[rows, cols].sum()

        unmatched = sorted(set(range(n_slots)) - set(rows))
        if unmatched:
            total = total - null_weight * slot_logp[unmatched, null_index].sum()

    return total, stats


def boundary_loss(
    start_logits: torch.Tensor,  # [B, W]
    end_logits: torch.Tensor,  # [B, W]
    word_mask: torch.Tensor,  # [B, W]
    examples,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    start_target = torch.zeros_like(start_logits)
    end_target = torch.zeros_like(end_logits)
    for b, ex in enumerate(examples):
        for span in ex.all_semantic_spans():
            if span.end <= start_logits.size(1):
                start_target[b, span.start] = 1.0
                end_target[b, span.end - 1] = 1.0

    mask = word_mask.to(start_logits.dtype)
    weight = torch.tensor(pos_weight, device=start_logits.device)
    loss_start = F.binary_cross_entropy_with_logits(
        start_logits.float(), start_target.float(), reduction="none", pos_weight=weight
    )
    loss_end = F.binary_cross_entropy_with_logits(
        end_logits.float(), end_target.float(), reduction="none", pos_weight=weight
    )
    denom = mask.sum().clamp(min=1.0)
    return ((loss_start + loss_end) * mask).sum() / denom


def match_events(
    type_logits: torch.Tensor,  # [K, T+1] for one example
    action_out: dict[str, torch.Tensor],
    component_out: dict[str, dict[str, torch.Tensor]],
    batch_idx: int,
    gold_events: list[GoldEvent],
    cost_weights: dict[str, float],
) -> list[tuple[int, int]]:
    """Hungarian assignment of predicted event slots to gold events."""
    if not gold_events:
        return []

    k_event = type_logits.size(0)
    type_logp = _log_softmax_last(type_logits)  # [K, T+1]

    action_start = _log_softmax_last(action_out["start_logits"][batch_idx])  # [K,1,W]
    action_end = _log_softmax_last(action_out["end_logits"][batch_idx])

    cost = type_logp.new_zeros((k_event, len(gold_events)))
    for j, gold in enumerate(gold_events):
        type_id = EVENT_TYPE_TO_ID[gold.event_type]
        cost[:, j] += cost_weights["type"] * (-type_logp[:, type_id])
        cost[:, j] += cost_weights["action"] * (
            -action_start[:, 0, gold.action_span.start]
            - action_end[:, 0, gold.action_span.end - 1]
        )
        for comp in TRIGGER_COMPONENTS:
            spans = gold.component_spans(comp)
            if not spans:
                continue
            head = component_out[comp]
            s_logp = _log_softmax_last(head["start_logits"][batch_idx])  # [K,S,W]
            e_logp = _log_softmax_last(head["end_logits"][batch_idx])
            span = spans[0]
            best = (-s_logp[:, :, span.start] - e_logp[:, :, span.end - 1]).min(dim=1).values
            cost[:, j] += cost_weights[comp] * best

    rows, cols = _assign(cost)
    return list(zip(rows, cols))


def event_type_loss(
    type_logits: torch.Tensor,  # [B, K, T+1]
    matches: list[list[tuple[int, int]]],
    examples,
) -> torch.Tensor:
    targets = torch.full(
        type_logits.shape[:2], NONE_EVENT_ID, dtype=torch.long, device=type_logits.device
    )
    for b, pairs in enumerate(matches):
        for slot, gold_idx in pairs:
            targets[b, slot] = EVENT_TYPE_TO_ID[examples[b].events[gold_idx].event_type]
    return F.cross_entropy(
        type_logits.float().flatten(0, 1), targets.flatten(0, 1), reduction="mean"
    )


def compute_total_loss(
    batch,
    outputs: dict,
    layout: RoleSlotLayout,
    weights: LossWeights,
    event_cost_weights: dict[str, float],
    prototype_anchor: torch.Tensor | None = None,
) -> LossBreakdown:
    """Assemble the full TARS objective for one batch."""
    event_out = outputs["event"]
    candidates = outputs["candidates"]
    arg_logits = outputs["arg_logits"]
    null_index = arg_logits.size(-1) - 1
    examples = batch.examples

    matches: list[list[tuple[int, int]]] = []
    for b, ex in enumerate(examples):
        matches.append(
            match_events(
                event_out.type_logits[b],
                event_out.action,
                event_out.components,
                b,
                ex.events,
                event_cost_weights,
            )
        )

    device = arg_logits.device
    l_type = event_type_loss(event_out.type_logits, matches, examples)
    l_action = torch.zeros((), device=device)
    l_components = torch.zeros((), device=device)
    l_arguments = torch.zeros((), device=device)

    num_matched = 0
    stats = {"gold_args": 0.0, "dropped_capacity": 0.0, "missing_candidate": 0.0,
             "dropped_component": 0.0}

    for b, pairs in enumerate(matches):
        for slot, gold_idx in pairs:
            gold = examples[b].events[gold_idx]
            num_matched += 1

            action_loss, _ = component_loss_for_slot(
                event_out.action, b, slot, [gold.action_span], nullable=False
            )
            l_action = l_action + action_loss

            for comp in TRIGGER_COMPONENTS:
                comp_loss, dropped = component_loss_for_slot(
                    event_out.components[comp], b, slot,
                    gold.component_spans(comp), nullable=True,
                )
                l_components = l_components + comp_loss
                stats["dropped_component"] += dropped

            arg_loss, arg_stats = argument_loss_for_event(
                arg_logits[b, slot],
                layout,
                gold,
                candidates.index_of[b],
                null_index,
                null_weight=weights.null_weight,
            )
            l_arguments = l_arguments + arg_loss
            for key, value in arg_stats.items():
                stats[key] += value

    # Event slots that matched nothing still train their NULL/NONE behaviour via
    # the type loss; their argument slots are supervised toward NULL so an
    # inactive slot cannot emit arguments.
    for b, pairs in enumerate(matches):
        matched_slots = {slot for slot, _ in pairs}
        for slot in range(arg_logits.size(1)):
            if slot in matched_slots:
                continue
            logp = _log_softmax_last(arg_logits[b, slot])
            l_arguments = l_arguments - weights.null_weight * logp[:, null_index].sum()

    denom = max(num_matched, 1)
    l_action = l_action / denom
    l_components = l_components / denom
    # Normalised per gold argument (DETR convention) so the ~75 NULL slots per
    # event cannot dilute the signal from the ~3 real arguments.
    l_arguments = l_arguments / max(stats["gold_args"], 1.0)

    l_boundary = boundary_loss(
        candidates.start_logits, candidates.end_logits, batch.word_mask, examples,
        pos_weight=weights.boundary_pos_weight,
    )
    l_proto = (
        prototype_anchor if prototype_anchor is not None else torch.zeros((), device=device)
    )

    total = (
        weights.event_type * l_type
        + weights.action * l_action
        + weights.tuple_components * l_components
        + weights.argument * l_arguments
        + weights.boundary * l_boundary
        + weights.prototype * l_proto
    )

    return LossBreakdown(
        total=total,
        parts={
            "event_type": l_type.detach(),
            "action": l_action.detach(),
            "components": l_components.detach(),
            "arguments": l_arguments.detach(),
            "boundary": l_boundary.detach(),
            "prototype": l_proto.detach(),
        },
        stats=stats,
    )
