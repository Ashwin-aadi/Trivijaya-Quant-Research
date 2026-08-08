from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalOutperformance(Strategy):
    rationale = (
        "Historical data suggests that certain sectors and stocks in the Indian market exhibit "
        "outperformance during specific times of the year. This strategy leverages technical indicators "
        "and macroeconomic conditions to predict favorable trading opportunities."
    )

    def __init__(self, window: int = 50, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute technical indicators
        history = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(window_size=self._window)).alias("ma"),
                ((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0)).alias("return"),
                ((pl.col("adj_close") - pl.col("adj_close").shift(self._window)) / pl.col("adj_close")).alias("rsi"),
                (
                    (pl.col("adj_close") > pl.col("adj_close").rolling_mean(window_size=self._window)).rank(
                        method="dense"
                    ).alias("rank")
                ),
            )
        )

        # Filter out symbols not in history
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate RSI and rank stocks based on combined scores
        rsi_scores = (
            history[symbols]
            .agg(
                (pl.col("return").mean().alias("avg_return")),
                (pl.col("rsi") > 0).sum().alias("overbought"),
                ((pl.col("ma") - pl.col("adj_close")) / pl.col("ma")).abs().mean().alias("volatility"),
            )
            .select(
                pl.all().rank(method="dense", descending=True),
                (pl.col("avg_return").rank(method="dense", descending=True) * 2)
                + (pl.col("overbought").rank(method="dense", descending=True))
                - (pl.col("volatility").rank(method="dense", descending=True)),
            )
        )

        # Get top N stocks based on the combined score
        rsi_scores = (
            rsi_scores.sort("combined_score", descending=True)
            .select(pl.all()[0])
            .to_series()
            .to_list()[: self._top_n]
        )
        picks = [symbol for symbol in view.symbols if any(symbol == s for s in rsi_scores)]

        # Assign weights
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest