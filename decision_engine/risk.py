"""Decision Engine v0 — entry/stop/risk derivation.

Locked spec: memory/decision_engine_v0_spec.md §Normalized event schema.

risk_pct = (entry_ref - stop_ref) / entry_ref
risk_atr = (entry_ref - stop_ref) / atr_14
extension_atr is source-supplied where available; nullable otherwise.
"""

from __future__ import annotations


def safe_div(num, denom):
    if num is None or denom is None:
        return None
    try:
        if denom == 0 or denom != denom:  # NaN check
            return None
        return float(num) / float(denom)
    except (TypeError, ValueError):
        return None


def derive_risk(*, entry_ref, stop_ref, atr):
    """Return (risk_pct, risk_atr) or (None, None) when inputs invalid.

    Long-direction only in v0. risk_pct positive when stop < entry; if stop >= entry
    we return None so rule 1/2 doesn't pretend a valid plan exists.
    """
    if entry_ref is None or stop_ref is None:
        return None, None
    try:
        e = float(entry_ref)
        s = float(stop_ref)
    except (TypeError, ValueError):
        return None, None
    if e != e or s != s or e <= 0:
        return None, None
    diff = e - s
    if diff <= 0:
        # stop above entry → not a valid long stop; treat as missing
        return None, None
    risk_pct = diff / e
    risk_atr = None
    if atr is not None:
        try:
            a = float(atr)
            if a == a and a > 0:
                risk_atr = diff / a
        except (TypeError, ValueError):
            pass
    return risk_pct, risk_atr


__all__ = ["derive_risk", "safe_div"]
