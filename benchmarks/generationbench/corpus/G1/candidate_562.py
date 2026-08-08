from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit stronger performance during specific times of the year. "
        "By identifying these seasonal trends, we can allocate capital towards assets that historically perform well at those times."
    )

    def __init__(self, window: int = 365, lookback_periods: int = 10) -> None:
        self._window = window
        self._lookback_periods = lookback_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_groups = (
            history.groupby("symbol")
            .agg(
                (pl.col("close").rolling_mean(window_size=self._lookback_periods).last().alias("mean_close")),
                (pl.col("close").rolling_corr(other="adj_close", window_size=self._lookback_periods).last().alias("corr")),
            )
            .sort("mean_close", descending=True)
            .head(self._lookback_periods)["symbol"]
        )

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in symbol_groups:
                continue
            closes = history.filter(pl.col("symbol") == symbol).select("close").to_numpy().flatten()
            recent_closes = [float(v) for v in closes[-self._lookback_periods:]]
            if all(c > 0 for c in recent_closes):
                weights[symbol] = 1.0 / len(symbol_groups)

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest