from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion occurs when a stock that has moved significantly from its mean "
        "level returns to that level. By comparing current prices with their trailing moving "
        "average, we can identify potential reversions."
    )

    def __init__(self, window: int = 60, zscore_threshold: float = 2.0) -> None:
        self._window = window
        self._zscore_threshold = zscore_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = {}
        for symbol in view.symbols:
            prices = history.select(
                pl.col("symbol").eq(symbol).alias("is_symbol"),
                pl.col("adj_close").alias("price"),
                pl.col("session_date"),
            )
            if not prices.height:
                continue
            zscores = (
                prices.sort(by="session_date")
                .group_by("is_symbol", maintain_order=True)
                .agg(
                    (pl.col("price") - pl.col("price").mean().over("is_symbol")) / 
                    pl.col("price").std().over("is_symbol").alias("zscore"),
                    (pl.col("session_date")).max().over("is_symbol").alias("last_session")
                )
            )
            latest_zscore = zscores.filter(pl.col("session_date") == zscores["last_session"]).select("zscore")
            if not latest_zscore.height:
                continue
            symbol_prices[symbol] = float(latest_zscore[0, 0])

        symbols_to_buy = [s for s in symbol_prices.keys() if abs(symbol_prices[s]) < self._zscore_threshold]
        weights = {s: 1.0 / len(symbols_to_buy) for s in symbols_to_buy}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest