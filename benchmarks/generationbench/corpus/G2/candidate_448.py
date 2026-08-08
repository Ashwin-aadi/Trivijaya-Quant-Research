from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency for asset prices to revert to their "
        "historical mean. In a market where recent prices are extreme, there is often a tendency "
        "for the price to move back towards its average level over a longer period."
    )

    def __init__(self, window: int = 60, mean_window: int = 20) -> None:
        self._window = window
        self._mean_window = mean_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = {s: float(v) for s, v in view.latest_close().items()}
        symbols = list(latest_closes.keys())

        means = (
            history.select(pl.col("adj_close").mean())
            .group_by("symbol")
            .collect()["adj_close"]
            .to_list()
        )
        recent_prices = [latest_closes[s] for s in symbols]

        deviations = [(p - m) / m for p, m in zip(recent_prices, means)]
        signals = sorted(zip(symbols, deviations), key=lambda x: abs(x[1]), reverse=True)

        if len(signals) < self._mean_window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s for s, d in signals[: self._mean_window]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest