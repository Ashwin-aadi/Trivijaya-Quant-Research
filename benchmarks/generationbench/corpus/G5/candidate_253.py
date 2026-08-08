from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices and financial returns are prone to revert "
        "to a long-term mean or average. For short horizons, deviations from the mean can be"
        " exploited for profits."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = float(closes.mean().to_series().item())
        mean_reversion_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            if abs(latest_close - mean_close) > 0.1 * mean_close:
                mean_reversion_signals[symbol] = 1.0

        weight = 1.0 / len(mean_reversion_signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in mean_reversion_signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest