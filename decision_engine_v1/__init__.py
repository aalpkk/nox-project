"""Decision Engine v1 — semantic label / context layer.

Spec: memory/decision_engine_v1_implementation_spec.md (LOCKED 2026-05-04 Tier 0,
+ additive REVISION 2026-05-04 Paper Signal Validity).
Design: memory/decision_engine_v1_design_pre_reg.md (LOCKED 2026-05-03).
Mapping: memory/decision_engine_v1_mapping_review.md §4 (CLOSED 2026-05-04).

Tier 0 ceiling: code scaffolding only. The runner stub at
tools/decision_engine_v1_run.py refuses to execute. Module functions are
importable but not invoked end-to-end at this tier.

Architectural separation (LOCKED):
  - Decision Engine v1 = label/context layer (this package).
  - paper_execution_v0 / paper_execution_v0_trigger_retest = paper-stream
    eligibility + validity authority (NOT this package).
  - portfolio_merge_paper = paper-stream merged reporting (NOT this package).
v1 reads paper-stream parquets read-only via paper_stream_link; v1 never
re-derives, never re-emits, never competes for source-of-truth.
"""

__version__ = "1.0.0-tier0"
__spec_lock_date__ = "2026-05-04"
__tier__ = 0
