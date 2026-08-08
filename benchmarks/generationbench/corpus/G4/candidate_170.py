from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where market trends are more pronounced "
        "during periods of high volatility. By scaling trade sizes based on historical "
        "volatility and ranking candidates based on their price relative to moving averages, "
        "the strategy aims to capture strong trends in volatile environments."
    )

    def __init__(self, window: int = 20, ma_window: int = 50, top_n: int = 30) -> None:
        self._window = window
        self._ma_window = ma_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._ma_window)

        if history.is_empty() or history.height < self._window + self._ma_window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate realized volatility
        volatility = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range"),
                ((pl.col("close") - pl.col("open")) ** 2).alias("price_diff"),
            )
            .with_column(
                (pl.col("adj_close").shift(-1) - pl.col("adj_close"))**2
                + pl.col("price_diff")
                / self._window,
                alias="volatility",
            )
            .group_by("symbol")
            .agg((pl.sum("volatility") ** 0.5).alias("historical_volatility"))
        )

        # Determine trade size based on volatility
        vol_rank = (
            history.with_columns(
                (pl.col("adj_close") - pl.col("adj_close").shift(self._ma_window))
                / pl.col("adj_close").shift(self._ma_window)
                .abs()
                .rank(method="dense", descending=True),
                alias="trend_strength",
            )
            .group_by("symbol")
            .agg(
                (pl.col("historical_volatility") * 0.5 + 1).alias("volatility_scaled_trend_strength"),
                pl.first("adj_close").alias("latest_close"),
                pl.first("close").alias("ma_close"),
            )
        )

        # Rank candidates based on trend strength and latest close relative to moving average
        ranking = (
            vol_rank.with_columns(
                (pl.col("latest_close") - pl.col("ma_close")) / pl.col("ma_close").abs()
                .rank(method="dense", descending=True),
                alias="relative_strength",
            )
            .sort(
                [pl.col("volatility_scaled_trend_strength"), pl.col("relative_strength")],
                descending=[True, True],
            )
            .head(self._top_n)
        )

        # Generate weights
        weight = 1.0 / len(ranking) if ranking.height > 0 else 0.0
        selected_symbols = [row["symbol"] for row in ranking.to_dicts()]

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest