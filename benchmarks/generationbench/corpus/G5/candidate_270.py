from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identify stocks with strong relative performance by comparing their "
        "recent price changes to the average of all NIFTY 100 constituents. "
        "Higher relative strength signals potential outperformance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbol_changes = {}
        nifty_100_avg_change = sum(
            float(v) - float(history["adj_close"].to_list()[i-1])
            for i, v in enumerate(history["adj_close"].to_list())
            if i >= self._window
        ) / (history.height - 1)

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            avg_change = sum(changes[-self._window:]) / self._window
            symbol_changes[symbol] = avg_change

        strongest_symbols = sorted(
            symbol_changes.items(), key=lambda x: (x[1] - nifty_100_avg_change), reverse=True
        )[:5]
        if not strongest_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strongest_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in strongest_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest