from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalAdjustment(Strategy):
    rationale = (
        "Seasonality in equity markets often stems from predictable changes in company "
        "performance driven by economic conditions. For instance, certain sectors may perform "
        "better during specific times of the year due to weather patterns or holiday seasons."
        "By identifying and adjusting for these seasonal effects, we can potentially capture "
        "anomalies in stock prices that result from such predictable variations."
    )

    def __init__(self, window: int = 365, min_periods: int = 10) -> None:
        self._window = window
        self._min_periods = min_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._min_periods:
            return Signal(information_available_at=stamp, weights={})

        seasonal_adjustments: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            closes = history.select([pl.col("session_date"), pl.col(symbol).alias("close")]).sort(
                "session_date"
            )
            if closes.height < self._min_periods:
                continue

            seasonal_adjustment = _calculate_seasonal_adj(closes, symbol)
            seasonal_adjustments[symbol] = seasonal_adjustment

        # Filter out symbols with no meaningful adjustment
        picks = [s for s in view.symbols if abs(seasonal_adjustments.get(s)) > 0.1]
        weight = 1.0 / len(picks) if picks else 0.0

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_seasonal_adj(history: pl.DataFrame, symbol: str) -> float:
    adjusted_close = history.select(pl.col("close"))
    mean_close = float(adjusted_close.mean().item())
    seasonal_factor = (float(history.filter(pl.col(symbol) > mean_close).shape[0]) / history.height)
    return seasonal_factor