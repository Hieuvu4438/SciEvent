"""Config loading and model construction.

Architectural capacities (``K_event``, ``max_span_width``, per-role slot counts)
are resolved from **TRAIN statistics only** and written into the run manifest so
a reviewer can see exactly which number came from where.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import yaml

from ..data.reader import train_statistics
from ..data.schema import SEMANTIC_ROLES, TRIGGER_COMPONENTS, WindowExample
from ..modeling.model import ModelConfig, TarsSciEventModel
from ..modeling.prototypes import encode_role_definitions
from .losses import LossWeights


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    parent = config.get("inherit")
    if parent:
        base_path = (Path(path).parent / parent).resolve()
        base = load_config(base_path)
        config = deep_merge(base, {k: v for k, v in config.items() if k != "inherit"})
    return config


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_capacities(
    train_examples: list[WindowExample], capacity_cfg: dict
) -> tuple[dict, dict]:
    """Derive capacities from TRAIN; returns ``(resolved, evidence)``."""
    stats = train_statistics(train_examples)
    evidence = {"train_statistics": stats, "rules": {}}

    k_event = capacity_cfg.get("k_event")
    if k_event is None:
        k_event = stats["max_events_per_window"] + 1
        cap = capacity_cfg.get("k_event_cap")
        if cap is not None and k_event > cap:
            evidence["rules"]["k_event"] = f"capped from {k_event} to {cap}"
            k_event = cap
        evidence["rules"].setdefault(
            "k_event", f"max_train_events_per_window({stats['max_events_per_window']}) + 1"
        )

    max_span_width = capacity_cfg.get("max_span_width")
    if max_span_width is None:
        p995 = stats["semantic_arg_width"]["p99_5"]
        floor = capacity_cfg.get("max_span_width_floor", 16)
        max_span_width = max(floor, min(64, int(math.ceil(p995))))
        evidence["rules"]["max_span_width"] = (
            f"min(64, ceil(train_p99_5={p995})) with floor {floor}"
        )

    role_mode = capacity_cfg.get("role_slot_mode", "max")
    role_counts = capacity_cfg.get("role_slot_counts")
    if not role_counts:
        source = (
            stats["role_multiplicity_max"]
            if role_mode == "max"
            else stats["role_multiplicity_p99"]
        )
        cap = capacity_cfg.get("role_slot_cap", 32)
        role_counts = {
            role: max(1, min(cap, int(source.get(role, 0)) + 1)) for role in SEMANTIC_ROLES
        }
        evidence["rules"]["role_slot_counts"] = (
            f"train role multiplicity {role_mode} + 1 (cap {cap})"
        )

    component_slots = capacity_cfg.get("component_slots")
    if not component_slots:
        source = stats["component_multiplicity_max"]
        component_slots = {
            c: max(1, int(source.get(c, 1))) for c in TRIGGER_COMPONENTS
        }
        evidence["rules"]["component_slots"] = "train component multiplicity max"

    resolved = {
        "k_event": int(k_event),
        "max_span_width": int(max_span_width),
        "role_slot_counts": {k: int(v) for k, v in role_counts.items()},
        "component_slots": {k: int(v) for k, v in component_slots.items()},
        "role_frequencies": stats["semantic_role_counts"],
    }
    return resolved, evidence


def load_role_definitions(path: str | Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    definitions = {r: raw["semantic_roles"][r]["definition"] for r in SEMANTIC_ROLES}
    return definitions


def build_model(
    config: dict,
    train_examples: list[WindowExample],
    device: torch.device,
) -> tuple[TarsSciEventModel, dict]:
    model_cfg = config["model"]
    resolved, evidence = resolve_capacities(train_examples, config.get("capacities", {}))

    definition_vectors = None
    prototype_mode = model_cfg.get("prototype_mode", "learned")
    if prototype_mode != "learned":
        from transformers import AutoModel

        definitions = load_role_definitions(config["data"]["role_definitions"])
        tokenizer_path = model_cfg["backbone_path"]
        from ..data.tokenizer_map import load_tokenizer

        tok = load_tokenizer(tokenizer_path)
        frozen = AutoModel.from_pretrained(tokenizer_path).to(device).eval()
        definition_vectors = encode_role_definitions(definitions, tok, frozen, device)
        del frozen
        torch.cuda.empty_cache()

    cfg = ModelConfig(
        backbone_path=model_cfg["backbone_path"],
        k_event=resolved["k_event"],
        decoder_layers=model_cfg.get("decoder_layers", 2),
        decoder_heads=model_cfg.get("decoder_heads", 8),
        dropout=model_cfg.get("dropout", 0.1),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
        attn_implementation=model_cfg.get("attn_implementation"),
        component_slots=resolved["component_slots"],
        role_slot_counts=resolved["role_slot_counts"],
        max_span_width=resolved["max_span_width"],
        max_action_width=model_cfg.get("max_action_width", 12),
        max_component_width=model_cfg.get("max_component_width", resolved["max_span_width"]),
        k_start=model_cfg.get("k_start", 64),
        k_end=model_cfg.get("k_end", 64),
        k_span=model_cfg.get("k_span", 512),
        candidate_dim=model_cfg.get("candidate_dim", 512),
        arg_proj_size=model_cfg.get("arg_proj_size", 512),
        use_discourse=model_cfg.get("use_discourse", False),
        discourse_layers=model_cfg.get("discourse_layers", 2),
        use_tuple_query=model_cfg.get("use_tuple_query", False),
        tuple_component_dim=model_cfg.get("tuple_component_dim", 256),
        use_action_distance=model_cfg.get("use_action_distance", True),
        prototype_mode=prototype_mode,
        prototype_kappa=model_cfg.get("prototype_kappa", 20.0),
        prototype_fixed_lambda=model_cfg.get("prototype_fixed_lambda", 0.5),
        prototype_anchor_weight=model_cfg.get("prototype_anchor_weight", 0.0),
        role_frequencies=resolved["role_frequencies"],
        null_bias=model_cfg.get("null_bias", 0.0),
        force_min_events=model_cfg.get("force_min_events", 1),
    )
    model = TarsSciEventModel(cfg, definition_vectors=definition_vectors)
    return model, {"resolved_capacities": resolved, "capacity_evidence": evidence}


def build_loss_weights(config: dict) -> LossWeights:
    weights = config.get("loss", {})
    return LossWeights(
        action=weights.get("lambda_action", 1.0),
        tuple_components=weights.get("lambda_tuple", 1.0),
        argument=weights.get("lambda_arg", 2.0),
        boundary=weights.get("lambda_bnd", 0.5),
        prototype=weights.get("lambda_proto", 0.05),
        event_type=weights.get("lambda_type", 1.0),
        null_weight=weights.get("null_weight", 0.1),
        boundary_pos_weight=weights.get("boundary_pos_weight", 5.0),
    )


def build_event_cost_weights(config: dict) -> dict[str, float]:
    cost = config.get("event_matching_cost", {})
    return {
        "type": cost.get("type", 2.0),
        "action": cost.get("action", 4.0),
        "Agent": cost.get("agent", 0.5),
        "PrimaryObject": cost.get("primary_object", 0.5),
        "SecondaryObject": cost.get("secondary_object", 0.25),
    }
