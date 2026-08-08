from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityBasedStrategy(Strategy):
    rationale = (
        "Seasonal effects can provide significant predictive power in equity markets. "
        "This strategy exploits the historical performance patterns observed during specific times of the year."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_df = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("close").mean().alias("avg_close")),
                (pl.col("close").shift(1).mean().alias("lagged_avg_close"))
            )
        )

        seasonal_signals: list[str] = []
        for symbol in view.symbols:
            if f"{symbol}_avg_close" not in seasonality_df.columns or f"{symbol}_lagged_avg_close" not in seasonality_df.columns:
                continue
            avg_close = float(seasonality_df[f"{symbol}_avg_close"].item())
            lagged_avg_close = float(seasonality_df[f"{symbol}_lagged_avg_close"].item())
            if (avg_close - lagged_avg_close) / lagged_avg_close > 0.1:
                seasonal_signals.append(symbol)

        weights: dict[str, float] = {s: 1.0 / len(seasonal_signals) for s in seasonal_signals} if seasonal_signals else {}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest