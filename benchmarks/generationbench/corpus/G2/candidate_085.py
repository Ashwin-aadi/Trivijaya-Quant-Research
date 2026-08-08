from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion2d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and financial returns will eventually reverse "
        "towards the mean. In a short horizon, extreme price movements can be expected to revert "
        "to historical norms."
    )

    def __init__(self, window: int = 2, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + 1)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.select(
            pl.col(pl.Utf8).to_dummies().mean().alias("mean_close")
        ).select("mean_close").item()
        deviations = [
            (float(v) - mean_close) / mean_close for v in closes["close"].to_list()[1:]
        ]
        
        signals: list[str] = []
        for symbol, deviation in zip(view.symbols, deviations):
            if abs(deviation) > self._threshold:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest