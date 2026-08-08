from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price levels revert to their mean over time. If a stock's price has recently "
        "fallen far below its trailing average, it is likely to bounce back towards the "
        "mean. This strategy buys such stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().column("adj_close")
        latest_closes = view.latest_close()

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in latest_closes.keys():
                continue
            adj_close = latest_closes[symbol]
            trailing_mean = mean_close.to_list()[-1]

            if adj_close < 0.9 * trailing_mean:
                signals[symbol] = 1.0 / len(signals)

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest