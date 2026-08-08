from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "Price reversion to the mean is a classical concept in finance. "
        "This strategy aims to identify stocks that have deviated significantly from their trailing average and are likely to revert."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        latest_closes: dict[str, float] = {s: float(v) for s, v in closes.to_dict().items()}

        avg_close = history["adj_close"].mean().to_list()[0]
        std_dev = history["adj_close"].std().to_list()[0]

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            z_score = (latest_closes[symbol] - avg_close) / std_dev
            if abs(z_score) > self._threshold:
                signals.append(symbol)

        weights: dict[str, float] = {s: 1.0 / len(signals) for s in signals}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest