from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion200d(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of stock prices to revert to their "
        "mean over longer time frames. By tracking against a 200-day moving average (MA), this "
        "strategy aims to identify stocks that have deviated significantly from their long-term "
        "trend, potentially offering entry points for profitable trades."
    )

    def __init__(self, window: int = 200, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

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
            ma = sum(values[-self._window:]) / self._window
            if values[-1] < 0.95 * ma or values[-1] > 1.05 * ma:
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