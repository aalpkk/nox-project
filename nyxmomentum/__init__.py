"""
nyxmomentum — BIST cross-sectional momentum ranking & portfolio construction.

Separate module. Not a copy of nyxalpha (single-strategy) or nyxexpansion
(event-driven continuation). This pipeline operates at rebalance granularity
(monthly v1, weekly v2) and ranks the tradeable universe by transferable
leadership quality.

Leakage contract, universe filters, label family, feature blocks, baselines,
portfolio construction, and cost-aware backtest live in sibling modules.
"""
