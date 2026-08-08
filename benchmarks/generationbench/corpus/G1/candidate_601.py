from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion seeks to capitalize on mean returns of asset prices over a short period. "
        "If an asset's price is significantly above or below its historical average, it is expected to revert."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbol = "NIFTY_100"  # Assuming we are interested in the NIFTY 100 constituent
        data = history.filter(pl.col("symbol") == symbol).select(
            pl.col("session_date"), pl.col("adj_close")
        )
        if data.is_empty():
            return Signal(information_available_at=stamp, weights={})

        adj_closes = [float(v) for v in data["adj_close"].to_list()]
        mean_price = sum(adj_closes) / len(adj_closes)
        z_score = (adj_closes[-1] - mean_price) / mean_price

        if abs(z_score) > 2:  # Triggering signal based on Z-score
            weights = {symbol: 1.0}
        else:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest