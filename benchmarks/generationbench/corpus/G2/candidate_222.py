from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Securities with high relative strength compared to their peers tend to continue "
        "outperforming over the next few trading days. This is based on the assumption that "
        "market participants may favor strong performers and underperformers may be undervalued."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)

        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each symbol
        returns = (
            (closes["adj_close"].shift(-1) / closes["adj_close"] - 1.0).alias("r")
        ).with_columns(closes[view.symbols].cumsum().over(view.symbols))

        # Rank symbols by their average return over the lookback period
        ranked = (
            returns.groupby("symbol").agg(
                pl.col("r").mean().alias("avg_return"),
                pl.col(view.symbols).mean().alias("mean_price")
            ).sort(pl.col("avg_return"), descending=True)
        )

        top_symbols = [s for s in view.symbols if s in ranked["symbol"].to_list()[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest