from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks with strong relative performance against the Nifty 100 index "
        "over a lookback period and allocates capital to these stocks, leveraging their tendency to continue outperforming."
    )

    def __init__(self, window: int = 180, top_n: int = 25) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_100_symbols = tuple(history["symbol"].to_list())
        if len(nifty_100_symbols) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        nifty_100_history = history.filter(pl.col("symbol").is_in(nifty_100_symbols))
        all_symbols = view.symbols

        # Calculate cumulative returns
        cum_returns = (
            nifty_100_history.groupby("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(240) - 1).alias("cum_return")
            )
            .sort("cum_return", descending=True)
            .head(self._top_n)
        )

        # Rank all symbols based on their relative strength
        rel_strength = (
            view.closes()
            .lazy()
            .join(nifty_100_history, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(240) - 1).alias("cum_return")
            )
            .group_by("symbol")
            .agg(
                (
                    (pl.col("cum_return") / cum_returns.select(pl.col("cum_return")).mean()).alias(
                        "rel_strength"
                    ),
                    pl.col("adj_close").max().alias("last_price"),
                )
            )
            .sort("rel_strength", descending=True)
            .head(self._top_n)
        ).collect()

        top_symbols = rel_strength["symbol"].to_list()[: self._top_n]
        weight = 4.0 / len(top_symbols)

        weights = {s: weight for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest