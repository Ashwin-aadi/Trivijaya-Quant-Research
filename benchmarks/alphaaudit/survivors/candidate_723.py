from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and reliability of price. "
        "Highly liquid stocks are more likely to maintain stable prices during "
        "market fluctuations. This strategy aims to equally weight highly liquid "
        "stocks, ensuring diversification while reducing the risk associated with "
        "less liquid securities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate volume over the lookback period
        history = (
            history.with_columns(
                (pl.col("volume").sum().over("symbol")).alias("total_volume")
            )
            .sort("total_volume", descending=True)
            .head(self._window)
        )

        symbols = [row["symbol"] for row in history.to_dicts()]
        if len(symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight each selected symbol
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest