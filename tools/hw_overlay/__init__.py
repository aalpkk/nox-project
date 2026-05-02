"""HW Overlay v1 — descriptive metric panel.

Spec: memory/hw_overlay_v1_spec.md (LOCKED 2026-05-02).

Pipeline:
  1. Scanner event adapters (this package) → unified roster
  2. tools/hw_overlay_v1_build.py → trade events parquet
  3. tools/hw_overlay_v1_metrics.py → metrics + random baseline
  4. tools/hw_overlay_v1_html.py → descriptive HTML panel

Phase A + low-N NOX: 6 scanners. alpha + regime_transition deferred (no
historical preds saved). HW source = tools/hwo_full_scan.py 1d 3y.
"""
