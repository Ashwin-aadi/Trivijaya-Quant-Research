from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts towards a trailing average. This can exploit mean-reverting behavior "
        "in stock prices, potentially leading to profitable trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        mean_price = sum(closes) / len(closes)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            recent_closes = [float(v) for v in history[symbol].drop_nulls().to_list()[-self._window :]]
            if all(c == mean_price for c in recent_closes):
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        weight_per_symbol = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp, weights=weight_per_symbol
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest