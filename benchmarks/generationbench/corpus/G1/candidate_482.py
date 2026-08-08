from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies stocks with recent trends based on volatility. High volatility "
        "indicates that the stock price is moving significantly, which can signal a strong trend. By "
        "buying such stocks, we aim to capture these trending moves."
    )

    def __init__(self, window: int = 20, std_dev_multiplier: float = 1.5) -> None:
        self._window = window
        self._std_dev_multiplier = std_dev_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.symbol.unique().to_list()]
        trend_scores: dict[str, float] = {}

        for symbol in symbols:
            window_data = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            closes = window_data.select("adj_close").to_series()
            rolling_mean = closes.rolling_mean(window=self._window)
            rolling_std_dev = closes.rolling_std(window=self._window)

            z_scores = (closes - rolling_mean) / pl.col("adj_close").shift(1).cast(float)
            trend_score = abs(z_scores.max()) * self._std_dev_multiplier
            trend_scores[symbol] = trend_score

        top_symbols = sorted(trend_scores, key=trend_scores.get, reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest