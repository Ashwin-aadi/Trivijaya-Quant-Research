from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedEquityStrategy(Strategy):
    rationale = (
        "This strategy integrates elements from earnings surprise, technical sentiment, momentum, and value factors "
        "to ensure a balanced approach that leverages both fundamental strength and short-term market dynamics."
    )

    def __init__(self, window: int = 50, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=1)
        history = view.history(lookback=self._window)

        if closes.height < 2 or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate Technical Sentiment (TS) and Momentum
        technical_sentiment = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") - pl.col("adj_close").shift(self._window)) / pl.col("adj_close").shift(self._window).alias("ts")
            )
            .group_by("symbol")
            .agg((pl.col("ts").mean().alias("avg_ts")))
            .sort("avg_ts", descending=True)
        )

        momentum = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") - pl.col("close").shift(1)) / pl.col("close").shift(1).alias("momentum")
            )
            .group_by("symbol")
            .agg((pl.col("momentum").mean().alias("avg_momentum")))
        )

        # Calculate Value Factor (VF)
        value_factor = (
            history.select(
                pl.col("symbol"),
                1 / pl.col("adj_close") / pl.col("book_value_per_share").alias("vf")
            )
            .group_by("symbol")
            .agg((pl.col("vf").mean().alias("avg_vf")))
        )

        # Combine scores
        combined_scores = (
            technical_sentiment.join(momentum, on="symbol", how="inner")
                .join(value_factor, on="symbol", how="inner")
                .select(
                    pl.all(),
                    (pl.col("avg_ts") + pl.col("avg_momentum") * 2 + pl.col("avg_vf")).alias("composite_score")
                )
        )

        # Filter top N stocks based on composite score
        top_stocks = combined_scores.sort("composite_score", descending=True).head(self._top_n)

        if top_stocks.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._top_n
        selected_symbols = [row["symbol"] for row in top_stocks.iter_rows()]

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest