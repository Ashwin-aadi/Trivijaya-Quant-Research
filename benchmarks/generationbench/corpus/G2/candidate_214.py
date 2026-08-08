from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in stock markets refers to patterns of price movements that occur regularly "
        "during a specific time of the year. These trends can be driven by various factors such as "
        "weather conditions, holidays, or seasonal economic activities. By identifying and exploiting "
        "these trends, one can capture abnormal returns."
    )

    def __init__(self, season_length: int = 90) -> None:
        self._season_length = season_length

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._season_length)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonal_strength = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_close_series = [float(v) for v in history[symbol].to_list()]
            if len(adj_close_series) < self._season_length:
                continue

            # Calculate rolling mean and standard deviation
            rolling_mean = pl.Series(adj_close_series).rolling_window(30).mean().to_list()
            rolling_std = pl.Series(adj_close_series).rolling_window(30).std().to_list()

            # Identify seasonal strength using the Z-score
            z_scores = [(v - m) / s if s != 0 else 0 for v, m, s in zip(adj_close_series[29:], rolling_mean[29:], rolling_std[29:])]
            seasonal_strength[symbol] = max(z_scores)

        # Select the top N symbols with highest seasonal strength
        picks = sorted(seasonal_strength.items(), key=lambda x: x[1], reverse=True)[:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol, _ in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest