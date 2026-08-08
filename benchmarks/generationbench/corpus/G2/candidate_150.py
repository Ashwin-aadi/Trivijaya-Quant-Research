from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with a higher relative strength (measured by their price returns compared to the "
        "NIFTY 100 index) tend to outperform over time. This strategy selects stocks that have "
        "recently outperformed the broader market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        index_closes = closes.select(pl.col("NIFTY_100_CLOSE")).to_dict(False)["NIFTY_100_CLOSE"]

        # Calculate daily returns for each stock and the NIFTY 100
        daily_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("stock_return"),
                (index_closes[pl.col("session_date")] / index_closes[pl.col("session_date").shift(1)] - 1.0)
                .alias("nifty_return"),
            )
            .filter(pl.col("session_date") < pl.col("session_date").max())
            .sort("symbol", "session_date")
        )

        # Compute relative strength
        rel_strengths = (
            daily_returns.groupby("symbol")
            .agg(
                (pl.col("stock_return") / pl.col("nifty_return")).alias("rel_strength"),
            )
            .filter(pl.col("rel_strength").is_not_null())
        )

        top_n_symbols = rel_strengths.sort("rel_strength", descending=True).select(
            "symbol"
        ).head(self._window)

        if top_n_symbols.height == 0:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())["session_date"][0]
    assert isinstance(newest, date)
    return newest