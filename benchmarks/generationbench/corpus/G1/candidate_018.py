from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks based on their relative strength "
        "against the broader market. Stocks that consistently outperform contribute more to "
        "the portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).sort("session_date")

        # Aggregate to get the window's return for each symbol and the market index
        window_returns = (
            history.groupby("symbol", maintain_order=True)
            .agg(
                [
                    (pl.col("return").sum().alias("total_return")),
                    pl.col("adj_close").last().alias("latest_price"),
                ]
            )
            .sort("total_return", descending=True)
        )

        # Get the latest closes of all symbols
        market_index = view.closes(lookback=self._window).transpose()
        market_index.columns = [date.isoformat(view.as_of)]

        # Calculate relative strength
        window_returns = (
            window_returns.join(
                market_index.select(pl.all().first()).with_columns(
                    (pl.col("adj_close").sum() / len(market_index.columns)).alias("market_value")
                ),
                on="symbol",
                how="left",
            )
            .with_columns(
                ((pl.col("total_return") - pl.col("market_value")) / pl.col("latest_price")).alias("relative_strength")
            )
            .sort("relative_strength", descending=True)
        )

        # Select top-performing symbols
        picks = window_returns.head(5)["symbol"].to_list()
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest