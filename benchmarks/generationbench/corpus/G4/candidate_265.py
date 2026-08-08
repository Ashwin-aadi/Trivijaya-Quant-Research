from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion60d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks that have "
        "deviated significantly from their historical price levels. It aims to profit from the "
        "tendency of stock prices to revert to their long-term averages, leveraging historical "
        "price patterns and market dynamics."
    )

    def __init__(self, lookback: int = 60, top_n: int = 20) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        sma_column = f"sma_{self._lookback}"
        std_dev_column = f"std_dev_{self._lookback}"

        sma = (
            closes.sort("session_date")
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").mean().alias(sma_column)),
                (pl.col("adj_close").std().alias(std_dev_column))
            )
        )

        z_scores: pl.DataFrame = (
            closes.join(sma, on="symbol")
            .with_columns((pl.col("adj_close") - pl.col(sma_column)) / pl.col(std_dev_column).alias("z_score"))
        )

        top_symbols = [str(symbol) for symbol in z_scores.sort("z_score", descending=False)["symbol"].to_list()[: self._top_n]]
        bottom_symbols = [str(symbol) for symbol in z_scores.sort("z_score")["symbol"].to_list()[: self._top_n]]

        weights: dict[str, float] = {s: 0.025 for s in top_symbols + bottom_symbols}

        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest