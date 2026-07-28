"""Low-volatility tilt within the NIFTY 100.

Low-volatility stocks have historically delivered better risk-adjusted returns than the broader
index, a pattern documented across many markets. This module ranks index-eligible constituents
by trailing realised volatility and holds the calmest names, on the expectation that this
premium is present in the Indian large-cap universe as well.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_LOOKBACK = 90
_MAX_NAMES = 15


def _eligible_symbols(index_membership: pl.DataFrame) -> list[str]:
    """Symbols that belong to the index today, per the constituent reference table."""
    current = index_membership.filter(pl.col("exit_date").is_null())
    return current["symbol"].to_list()


def _low_vol_ranking(frame: pl.DataFrame) -> pl.DataFrame:
    """Rank symbols by trailing realised volatility, ascending."""
    ordered = frame.sort(["symbol", "session_date"])
    returns = ordered.with_columns(pl.col("adj_close").pct_change().over("symbol").alias("ret"))
    return returns.group_by("symbol").agg(volatility=pl.col("ret").std())


class LowVolatilityTilt(Strategy):
    """Holds the calmest names among the index's constituents, ranked by trailing volatility."""

    rationale = (
        "Low-volatility stocks have earned better risk-adjusted returns than the broader market "
        "across many studies, a pattern that appears to persist in Indian large caps as well. "
        "Ranking eligible constituents by trailing realised volatility and holding the calmest "
        "names should capture that premium while staying within investable, index-eligible "
        "names."
    )

    def __init__(self, index_membership: pl.DataFrame) -> None:
        # The reference table records each constituent's entry and exit dates; names that have
        # since left the index are excluded so the book only ever holds investable constituents.
        self._eligible = _eligible_symbols(index_membership)

    def generate(self, view: MarketView) -> Signal:
        frame = view.history(lookback=_LOOKBACK).filter(pl.col("symbol").is_in(self._eligible))
        if frame.is_empty():
            return Signal(information_available_at=view.as_of, weights={})
        ranked = _low_vol_ranking(frame).drop_nulls("volatility")
        ranked = ranked.sort("volatility").head(_MAX_NAMES)
        names = ranked["symbol"].to_list()
        if not names:
            return Signal(information_available_at=view.as_of, weights={})
        weight = 1.0 / len(names)
        return Signal(information_available_at=view.as_of, weights={s: weight for s in names})
