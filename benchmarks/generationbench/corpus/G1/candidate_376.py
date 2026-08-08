from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when a stock returns to its mean after deviating significantly. "
        "By identifying stocks that have moved far from their 20-day average and are now close to it, "
        "we can capture potential mean-reverting behavior."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        avg_close = (history["adj_close"] / history["adj_close"].shift(self._window)).mean().item()
        std_dev = (history["adj_close"] - avg_close).std().item()

        reversion_cands: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in view.closes(lookback=self._window):
                continue
            recent_closes = [float(v) for v in history[symbol].to_list()[-self._window:]]
            mean_close = sum(recent_closes) / self._window
            z_score = (recent_closes[-1] - mean_close) / std_dev

            if abs(z_score) > 2:
                reversion_cands.append(symbol)

        weights = {s: 0.5 for s in reversion_cands}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest