from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Dispersion20d(Strategy):
    rationale = (
        "By measuring the daily closing price deviation from a 20-day simple moving average, "
        "this strategy captures recent momentum and mean reversion tendencies. Deviations from "
        "the moving average indicate potential shifts towards dispersion or range compression in "
        "the Indian market."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma_values = []
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window:
                continue

            sma20 = sum(close_series[-self._window:]) / self._window
            deviation = (close_series[-1] - sma20) / sma20
            sma_values.append(deviation)

        sma_values.sort(reverse=True)
        top_deviations = sma_values[: self._top_n]
        for symbol, deviation in zip(view.symbols, sma_values):
            if deviation >= max(top_deviations):
                picks.append(symbol)

        picks = picks[: self._top_n]
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