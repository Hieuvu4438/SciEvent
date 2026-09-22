"""Phase 9 oracle diagnostics (O1-O5).

    O1  normal: predicted type + predicted trigger tuple + predicted arguments
    O2  GOLD semantic argument spans -> classify role only
    O3  GOLD event type              -> predicted trigger/arguments
    O4  GOLD trigger tuple           -> predicted semantic arguments
    O5  GOLD type + GOLD tuple       -> predicted semantic arguments

O2 bounds the role-classification ceiling given perfect boundaries. O4 shows
how much argument performance is lost through trigger conditioning.

**Oracle results never enter a results table.** They are run on train/dev only.
"""

from __future__ import annotations

from dataclasses import dataclass

ORACLE_MODES: dict[str, dict[str, bool]] = {
    "O1": {},
    "O2": {"gold_spans": True},
    "O3": {"gold_event_type": True},
    "O4": {"gold_tuple": True},
    "O5": {"gold_event_type": True, "gold_tuple": True},
}

ORACLE_DESCRIPTIONS = {
    "O1": "predicted type + predicted trigger tuple + predicted arguments",
    "O2": "GOLD semantic argument spans -> role classification only",
    "O3": "GOLD event type -> predicted trigger/arguments",
    "O4": "GOLD trigger tuple -> predicted semantic arguments",
    "O5": "GOLD type + GOLD tuple -> predicted semantic arguments",
}


@dataclass
class OracleApplicability:
    mode: str
    applicable: bool
    reason: str


def applicability(mode: str, model_config) -> OracleApplicability:
    """O4/O5 only alter the argument path when tuple conditioning is enabled.

    In H0 the argument decoder is conditioned on the raw event-slot state, so
    pinning the trigger tuple to gold cannot change the semantic arguments.
    Reporting O4 as informative for H0 would be misleading.
    """
    if mode in ("O4", "O5") and not getattr(model_config, "use_tuple_query", False):
        effect = "event-type decoding only" if mode == "O5" else "nothing"
        return OracleApplicability(
            mode=mode,
            applicable=False,
            reason=(
                "argument conditioning does not use the trigger tuple "
                f"(use_tuple_query=false), so this oracle affects {effect}"
            ),
        )
    return OracleApplicability(mode=mode, applicable=True, reason="")
