from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to provide stable returns over time. By allocating "
        "more capital to these stocks, the overall portfolio risk can be reduced while still "
        "pursuing positive returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if len(symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the rolling standard deviation
        volatility = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean()
            )
            .group_by("symbol")
            .agg(pl.Series.to_list())
            .with_column(
                pl.concat_list([pl.lit(v) for v in history["session_date"].to_list()])
                .alias("dates")
            )
        )

        # Calculate the mean of rolling standard deviation
        volatility = (
            volatility.join(
                history.select(pl.col("symbol"), pl.col("adj_close")),
                on="symbol",
                how="inner",
            )
            .with_columns(
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean().alias("volatility")
            )
        )

        volatility = (
            volatility.select(
                "symbol", pl.col("volatility").rank(method="average", descending=True).alias("rank")
            )
        )

        # Get the top low-volatility symbols
        top_symbols = [row["symbol"] for row in volatility.to_dicts()[: self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest