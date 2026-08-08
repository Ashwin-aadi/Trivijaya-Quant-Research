from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Prices that revert to mean levels after diverging tend to provide profitable "
        "opportunities. By identifying symbols where the current price is far from its trailing "
        "mean, we can make informed bets on a reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("trailing_mean"))
            .to_pandas()["trailing_mean"]
        )
        
        symbols = []
        for symbol in view.symbols:
            if symbol not in mean_close.index:
                continue
            latest_close = view.latest_close()[symbol]
            z_score = (latest_close - mean_close[symbol]) / mean_close[symbol]
            if abs(z_score) > 2.0:  # Select symbols with large z-scores indicating reversion potential
                symbols.append(symbol)

        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest