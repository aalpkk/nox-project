"""ema_context — descriptive EMA feature health layer.

Phase 0 spec: memory/ema_context_phase0_spec.md (LOCKED 2026-05-03).
Descriptive only. No edge claim, no gate, no model, no AL/SAT.
"""
from ema_context.features import compute_ema_features, EMA_PERIODS
from ema_context.tags import assign_tags, fit_breakpoints, BREAKPOINTS_VERSION

__all__ = [
    "compute_ema_features",
    "EMA_PERIODS",
    "assign_tags",
    "fit_breakpoints",
    "BREAKPOINTS_VERSION",
]
