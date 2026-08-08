from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts towards historical means; buy undervalued stocks and sell overvalued ones "
        "based on a trailing reference price level."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(pl.col("adj_close").mean().alias("mean")).to_series().to_list()[0]
        std_close = history.select(pl.col("adj_close").std()).to_series().to_list()[0]

        symbols_with_price = [
            (symbol, float(price))
            for symbol, price in view.latest_close().items()
            if symbol in history.columns
        ]

        weights: dict[str, float] = {}
        for symbol, latest_close in symbols_with_price:
            trailing_mean = (
                history.select(pl.col(symbol).mean()).to_series().to_list()[0]
            )
            z_score = (latest_close - trailing_mean) / std_close if std_close > 0 else 0
            weights[symbol] = max(0.1, min(0.9, 1 + z_score))

        return Signal(
            information_available_at=stamp,
            weights={k: v for k, v in weights.items() if v > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest