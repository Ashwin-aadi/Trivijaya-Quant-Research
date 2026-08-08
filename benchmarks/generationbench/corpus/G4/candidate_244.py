from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion50d(Strategy):
    rationale = (
        "This strategy exploits mean reversion by identifying stocks whose prices have "
        "deviated significantly from their 50-day simple moving average (SMA). It aims to "
        "capitalize on temporary price anomalies and benefit from the natural tendency of "
        "prices to return to historical norms."
    )

    def __init__(self, window: int = 50, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma = (
            history.group_by("symbol")
            .agg(
                (pl.col("close").mean().alias("sma"))
            )
            .with_columns((pl.col("close") - pl.col("sma")).abs().alias("diff"))
        )

        top_buy_symbols = [
            symbol for symbol, diff in sma.sort("diff", descending=True)["symbol"].to_list()[: self._top_n]
        ]
        top_sell_symbols = [
            symbol for symbol, diff in sma.sort("diff", descending=False)["symbol"].to_list()[: self._top_n]
        ]

        buy_weights = {s: 1.0 / len(top_buy_symbols) for s in top_buy_symbols}
        sell_weights = {s: -1.0 / len(top_sell_symbols) for s in top_sell_symbols}

        weights = {**buy_weights, **sell_weights}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest