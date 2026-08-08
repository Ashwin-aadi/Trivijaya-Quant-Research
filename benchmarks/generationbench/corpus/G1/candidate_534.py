from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the strongest relative performance against the broader market "
        "indicates that these stocks are outperforming and could continue to do so."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate the returns
        price_returns = (
            history.select(
                pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .group_by("symbol", maintain_order=True)
            .agg(pl.col("r").mean().alias("avg_return"))
        )

        # Calculate the market return
        market_returns = history.select(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("market_return")
        ).select([pl.col("market_return").mean().alias("market_avg_return")])

        # Merge price and market returns
        merged = price_returns.join(market_returns, on="symbol", how="left")
        if merged.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate relative strength
        merged = merged.with_columns(
            (pl.col("avg_return") / pl.col("market_avg_return")).alias("relative_strength")
        )

        # Get top N symbols by relative strength
        top_symbols = (
            merged.sort("relative_strength", descending=True)
            .select(pl.col("symbol"))
            .head(self._window)["symbol"]
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest