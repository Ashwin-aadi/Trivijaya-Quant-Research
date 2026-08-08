from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion2d(Strategy):
    rationale = (
        "Mean reversion identifies stocks that have deviated significantly from their historical "
        "mean and are likely to return to it. In a short horizon, this can lead to profitable "
        "trades."
    )

    def __init__(self, window: int = 2, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = sum(values) / len(values)
            z_score = (values[-1] - mean_close) / (sum((v - mean_close) ** 2 for v in values) ** 0.5)

            if abs(z_score) > self._threshold:
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest