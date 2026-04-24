"""nyxexpansion timing-clean retention stage.

Locked 2026-04-25 — recomputes candidate features with T's daily bar truncated
at 17:00 TR so the live scan at 17:30 is not contaminated by end-of-session
volume surge. Scored by a persisted LightGBM surrogate of the v4C in-memory
regressor; candidates must rank ≤ 10 within the day's competitor panel to
pass the gate.

See memory/nyxexp_timing_clean_retention_filter_locked.md for the locked rule
and memory/nyxexp_retention_filter_impl_plan.md for the implementation plan.
"""
