"""Hold names trading below the turnover-weighted average price of the recent window."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible


class PriceVsVwap(Strategy):
    """Compares the last close against a rupee-weighted average of the window's closes."""

    rationale = (
        "Weighting each session's close by the rupees that changed hands that session gives the "
        "price at which the average unit of money in the window transacted. A name trading below "
        "that level is one where the typical buyer of the period is underwater, so the rule is a "
        "reversion tilt toward names the flow has already paid more for."
    )

    def __init__(self, window: int = 63) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_close", "turnover_inr"]
            ).sort("session_date")
            if rows.height < self._window:
                continue
            closes = [float(v) for v in rows["adj_close"].to_list()]
            traded = [float(v) for v in rows["turnover_inr"].to_list()]
            total = sum(traded)
            if total <= 0:
                continue
            vwap = sum(c * t for c, t in zip(closes, traded, strict=True)) / total
            if closes[-1] < vwap:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
