from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines the relative strength of a stock with its recent volatility "
        "to identify stocks that are both trending strongly and not experiencing excessive price "
        "fluctuations. Such stocks are considered robust and potentially reliable for investment."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate relative strength
        relative_strength: pl.DataFrame = (
            closes.pipe(lambda df: df.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            ))
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.col("r").mean().alias("avg_return"))
        )

        # Calculate volatility
        volatility: pl.DataFrame = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").std()).alias("volatility"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .sort("session_date", descending=True)
        )

        # Combine relative strength and volatility
        combined = (
            history.join(relative_strength, on="symbol", how="inner")
            .join(volatility, on="symbol", how="inner")
        )

        ranked_symbols = (
            combined.select(["symbol", "avg_return", "volatility"])
            .sort("avg_return", descending=True)
            .group_by("symbol")
            .agg(
                (pl.col("avg_return") / pl.col("volatility").rank(method="dense", descending=True)).alias("score"),
                pl.col("avg_return").sum().alias("total_avg_return"),
                pl.col("volatility").mean().alias("mean_volatility")
            )
        )

        # Select top N symbols based on the score
        selected_symbols = (
            ranked_symbols.sort("score", descending=True)
            .select("symbol")
            .head(self._top_n)
            .to_list()
        )

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest