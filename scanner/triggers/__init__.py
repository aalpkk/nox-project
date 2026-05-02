"""Per-family trigger functions. Each module exposes:

    detect(df: pd.DataFrame, *, asof: pd.Timestamp | None = None) -> list[dict]

where each dict already carries the schema columns for that family's row
(identity + audit + contract + common__ + family__). Score columns are
populated downstream by `scanner.scoring`.

`df` is daily-indexed for daily-class families (horizontal_base), or 1h-indexed
for 1h-class families (mitigation_block, breaker_block). The engine routes
panels per family based on its declared frequency.
"""
from . import breaker_block, horizontal_base, mitigation_block

__all__ = ["horizontal_base", "mitigation_block", "breaker_block"]
