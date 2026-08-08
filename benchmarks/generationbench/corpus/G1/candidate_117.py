from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting towards low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        symbol_groups = history.group_by("symbol").agg(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        )
        returns_mean = symbol_groups.select(
            pl.col("returns").mean().alias("mean_return")
        ).collect()
        std_dev = history.group_by("symbol").agg(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).std().alias("volatility")
        ).collect()

        combined_data = returns_mean.join(std_dev, on="symbol", how="inner")
        sorted_symbols = (
            combined_data.with_columns(
                (pl.col("volatility") / pl.col("mean_return").abs()).alias("tilted_score")
            )
            .sort("tilted_score", descending=True)
            .select(pl.col("symbol"))
        ).to_series()

        top_n_symbols = [str(symbol) for symbol in sorted_symbols[:10]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest