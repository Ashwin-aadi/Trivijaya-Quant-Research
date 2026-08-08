from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and can indicate the ease of trading. "
        "This strategy screens for symbols with high liquidity (volume) and "
        "allocates equal weights to them."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.groupby("symbol")
                   .agg(
                       pl.col("volume").mean().alias("avg_volume"),
                       pl.col("adj_close").last().alias("latest_close"),
                   )
                   .sort("avg_volume", descending=True)
                   .head(self._top_n)  # Adjust the number of top symbols as needed
        )

        if liquidity_screened.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in liquidity_screened.to_dicts()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest