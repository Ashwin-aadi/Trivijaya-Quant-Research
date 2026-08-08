from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and returns tend to revert to a long-term "
        "mean. In the short term, significant deviations from this mean can be exploited for "
        "profitability. For stocks with high volatility, such deviations are more likely to occur."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = sum(values) / len(values)
            recent_close = values[-1]
            z_score = (recent_close - mean_close) / pl.col("adj_close").std().over(window=self._window).fill_none(0)[symbol]

            if abs(z_score) > 2.0:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest