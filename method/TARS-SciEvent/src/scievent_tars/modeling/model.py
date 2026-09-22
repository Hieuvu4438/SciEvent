"""TARS-SciEvent model assembly.

Component switches (all off in H0):

* ``use_discourse``    -- H4 rhetorical sentence hierarchy
* ``use_tuple_query``  -- H1 compositional Agent-Action-Object tuple anchoring
* ``prototype_mode``   -- H2 (``fixed``) / H3 (``adaptive``) role prototypes

H0 therefore is: ModernBERT-large + window-level event-set decoder +
Action/Agent/PO/SO heads + candidate semantic spans + simple learned role
representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..data.collator import Batch
from ..data.schema import (
    EVENT_TYPES,
    NONE_EVENT_ID,
    SEMANTIC_ROLES,
    TRIGGER_COMPONENTS,
    Span,
)
from .argument_set import ArgumentSetDecoder, RoleSlotLayout
from .discourse import SentenceHierarchy
from .encoder import WordEncoder
from .event_set import EventSetDecoder
from .prototypes import RolePrototypes
from .span_candidates import SpanCandidateModule
from .tuple_query import TupleQueryBuilder


@dataclass
class PredictedEvent:
    event_type: str
    action_span: Span
    components: dict[str, list[Span]] = field(default_factory=dict)
    arguments: list[tuple[str, Span]] = field(default_factory=list)
    type_score: float = 0.0


@dataclass
class ModelConfig:
    backbone_path: str
    k_event: int = 2
    decoder_layers: int = 2
    decoder_heads: int = 8
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    attn_implementation: str | None = None

    component_slots: dict[str, int] = field(
        default_factory=lambda: {"Agent": 1, "PrimaryObject": 1, "SecondaryObject": 4}
    )
    role_slot_counts: dict[str, int] = field(default_factory=dict)

    max_span_width: int = 40
    max_action_width: int = 12
    max_component_width: int = 40
    k_start: int = 64
    k_end: int = 64
    k_span: int = 512
    candidate_dim: int = 512
    arg_proj_size: int = 512

    use_discourse: bool = False
    discourse_layers: int = 2
    use_tuple_query: bool = False
    tuple_component_dim: int = 256
    use_action_distance: bool = True

    prototype_mode: str = "learned"
    prototype_kappa: float = 20.0
    prototype_fixed_lambda: float = 0.5
    prototype_anchor_weight: float = 0.0
    role_frequencies: dict[str, int] = field(default_factory=dict)

    null_bias: float = 0.0
    force_min_events: int = 1


class TarsSciEventModel(nn.Module):
    def __init__(self, config: ModelConfig, definition_vectors: torch.Tensor | None = None):
        super().__init__()
        self.config = config
        self.encoder = WordEncoder(
            config.backbone_path,
            gradient_checkpointing=config.gradient_checkpointing,
            dropout=config.dropout,
            attn_implementation=config.attn_implementation,
        )
        hidden = self.encoder.hidden_size

        self.discourse = (
            SentenceHierarchy(hidden, num_layers=config.discourse_layers,
                              num_heads=config.decoder_heads, dropout=config.dropout)
            if config.use_discourse
            else None
        )
        self.event_decoder = EventSetDecoder(
            hidden,
            k_event=config.k_event,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.dropout,
            component_slots=config.component_slots,
        )
        self.tuple_builder = (
            TupleQueryBuilder(hidden, component_dim=config.tuple_component_dim,
                              dropout=config.dropout)
            if config.use_tuple_query
            else None
        )
        self.candidates = SpanCandidateModule(
            hidden,
            max_span_width=config.max_span_width,
            k_start=config.k_start,
            k_end=config.k_end,
            k_span=config.k_span,
            out_dim=config.candidate_dim,
            dropout=config.dropout,
        )
        self.prototypes = RolePrototypes(
            hidden,
            mode=config.prototype_mode,
            definition_vectors=definition_vectors,
            role_frequencies=config.role_frequencies,
            kappa=config.prototype_kappa,
            fixed_lambda=config.prototype_fixed_lambda,
            anchor_weight=config.prototype_anchor_weight,
        )
        self.layout = RoleSlotLayout.build(config.role_slot_counts)
        self.arg_decoder = ArgumentSetDecoder(
            hidden,
            candidate_dim=config.candidate_dim,
            layout=self.layout,
            proj_size=config.arg_proj_size,
            dropout=config.dropout,
            use_action_distance=config.use_action_distance,
        )

    # -- forward ----------------------------------------------------------------

    def forward(self, batch: Batch, training: bool, oracle: dict | None = None) -> dict:
        """``oracle`` enables the Phase 9 diagnostics (O2-O5).

        Oracle runs replace part of the prediction with gold and therefore never
        enter any results table; they only bound what the rest of the model
        could achieve.
        """
        oracle = oracle or {}
        word_states = self.encoder(
            batch.input_ids,
            batch.attention_mask,
            batch.piece_to_word,
            batch.word_mask.size(1),
        )
        if self.discourse is not None:
            word_states = self.discourse(
                word_states, batch.word_mask, batch.word_sentence_ids
            )

        event_out = self.event_decoder(word_states, batch.word_mask)

        if oracle.get("gold_tuple"):
            self._apply_gold_tuple(event_out, batch)

        forced = None
        if training or oracle.get("gold_spans"):
            forced = [
                [sp.as_tuple() for sp in ex.all_semantic_spans()] for ex in batch.examples
            ]
        candidates = self.candidates(
            word_states,
            batch.word_mask,
            batch.num_words,
            forced_spans=forced,
            only_forced=bool(oracle.get("gold_spans")),
        )

        condition = event_out.queries
        if self.tuple_builder is not None:
            condition = self.tuple_builder(
                event_out.queries,
                event_out.type_logits,
                event_out.action,
                event_out.components,
                word_states,
            )

        arg_logits = self.arg_decoder(
            condition,
            self.prototypes(),
            candidates.features,
            candidates.mask,
            candidates.spans,
            batch.word_sentence_ids,
            event_out.action["start_logits"],
            null_bias=0.0 if training else self.config.null_bias,
        )

        return {
            "word_states": word_states,
            "event": event_out,
            "candidates": candidates,
            "arg_logits": arg_logits,
            "condition": condition,
        }

    @torch.no_grad()
    def _apply_gold_tuple(self, event_out, batch: Batch) -> None:
        """O4/O5: pin Action/Agent/PO/SO to their gold spans (one-hot logits)."""
        peak = 20.0
        for b, example in enumerate(batch.examples):
            if not example.events:
                continue
            gold = example.events[0]
            for name, spans, head in (
                ("Action", [gold.action_span], event_out.action),
                ("Agent", gold.agent_spans, event_out.components["Agent"]),
                ("PrimaryObject", gold.primary_object_spans,
                 event_out.components["PrimaryObject"]),
                ("SecondaryObject", gold.secondary_object_spans,
                 event_out.components["SecondaryObject"]),
            ):
                n_slots = head["start_logits"].size(2)
                for slot in range(n_slots):
                    if slot < len(spans):
                        span = spans[slot]
                        head["start_logits"][b, :, slot] = 0.0
                        head["end_logits"][b, :, slot] = 0.0
                        head["start_logits"][b, :, slot, span.start] = peak
                        head["end_logits"][b, :, slot, span.end - 1] = peak
                        if "presence_logits" in head:
                            head["presence_logits"][b, :, slot] = peak
                    elif "presence_logits" in head:
                        head["presence_logits"][b, :, slot] = -peak

    # -- decoding ---------------------------------------------------------------

    @staticmethod
    def _best_span(
        start_logp: torch.Tensor, end_logp: torch.Tensor, num_words: int, max_width: int
    ) -> tuple[Span, float]:
        s = start_logp[:num_words]
        e = end_logp[:num_words]
        grid = s.unsqueeze(1) + e.unsqueeze(0)  # [W, W] over (start, inclusive end)
        positions = torch.arange(num_words, device=s.device)
        offset = positions.unsqueeze(0) - positions.unsqueeze(1)
        grid = grid.masked_fill((offset < 0) | (offset >= max_width), float("-inf"))
        flat = int(grid.flatten().argmax())
        start, end_incl = divmod(flat, num_words)
        return Span(start, end_incl + 1), float(grid.flatten()[flat])

    @torch.no_grad()
    def decode(self, batch: Batch, outputs: dict,
               oracle: dict | None = None) -> list[list[PredictedEvent]]:
        cfg = self.config
        oracle = oracle or {}
        event_out = outputs["event"]
        candidates = outputs["candidates"]
        arg_logits = outputs["arg_logits"]
        null_index = arg_logits.size(-1) - 1

        type_logp = torch.log_softmax(event_out.type_logits.float(), dim=-1)
        action_start = torch.log_softmax(event_out.action["start_logits"].float(), dim=-1)
        action_end = torch.log_softmax(event_out.action["end_logits"].float(), dim=-1)

        results: list[list[PredictedEvent]] = []
        for b in range(len(batch)):
            num_words = int(batch.num_words[b])
            slot_types = type_logp[b].argmax(dim=-1).tolist()
            active = [k for k, t in enumerate(slot_types) if t != NONE_EVENT_ID]

            if not active and cfg.force_min_events > 0:
                # Every TRAIN window contains at least one event; this prior is
                # train-derived and never uses the gold count of this example.
                best_slots = (
                    type_logp[b, :, :NONE_EVENT_ID].max(dim=-1).values
                    - type_logp[b, :, NONE_EVENT_ID]
                ).topk(min(cfg.force_min_events, cfg.k_event)).indices.tolist()
                active = best_slots
                slot_types = [
                    int(type_logp[b, k, :NONE_EVENT_ID].argmax()) if k in active else t
                    for k, t in enumerate(slot_types)
                ]

            events: list[PredictedEvent] = []
            for k in active:
                type_id = slot_types[k]
                if type_id == NONE_EVENT_ID:
                    type_id = int(type_logp[b, k, :NONE_EVENT_ID].argmax())
                if oracle.get("gold_event_type") and batch.examples[b].events:
                    from ..data.schema import EVENT_TYPE_TO_ID

                    type_id = EVENT_TYPE_TO_ID[batch.examples[b].events[0].event_type]
                action_span, _ = self._best_span(
                    action_start[b, k, 0], action_end[b, k, 0], num_words, cfg.max_action_width
                )

                components: dict[str, list[Span]] = {}
                for comp in TRIGGER_COMPONENTS:
                    head = event_out.components[comp]
                    presence = head["presence_logits"][b, k]
                    s_logp = torch.log_softmax(head["start_logits"][b, k].float(), dim=-1)
                    e_logp = torch.log_softmax(head["end_logits"][b, k].float(), dim=-1)
                    spans: list[Span] = []
                    for slot in range(presence.size(0)):
                        if float(presence[slot]) <= 0:
                            continue
                        span, _ = self._best_span(
                            s_logp[slot], e_logp[slot], num_words, cfg.max_component_width
                        )
                        spans.append(span)
                    components[comp] = spans

                choices = arg_logits[b, k].argmax(dim=-1).tolist()
                seen: set[tuple[str, int, int]] = set()
                arguments: list[tuple[str, Span]] = []
                for flat_slot, choice in enumerate(choices):
                    if choice == null_index:
                        continue
                    if not bool(candidates.mask[b, choice]):
                        continue
                    role = SEMANTIC_ROLES[self.layout.role_index[flat_slot]]
                    start, end = candidates.spans[b, choice].tolist()
                    key = (role, start, end)
                    if key in seen:  # duplicates within the same event+role
                        continue
                    seen.add(key)
                    arguments.append((role, Span(start, end)))

                events.append(
                    PredictedEvent(
                        event_type=EVENT_TYPES[type_id],
                        action_span=action_span,
                        components=components,
                        arguments=arguments,
                        type_score=float(type_logp[b, k, type_id]),
                    )
                )
            results.append(events)
        return results

    # -- parameter groups -------------------------------------------------------

    def parameter_groups(self, backbone_lr: float, head_lr: float, weight_decay: float):
        decay, no_decay = [], []
        backbone_decay, backbone_no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("encoder.backbone")
            skip_decay = param.ndim <= 1 or name.endswith(".bias")
            target = (
                (backbone_no_decay if skip_decay else backbone_decay)
                if is_backbone
                else (no_decay if skip_decay else decay)
            )
            target.append(param)
        return [
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]
