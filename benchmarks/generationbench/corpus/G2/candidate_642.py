from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength compared to the broader market tend to outperform "
        "over time. This is based on the idea that strong stocks can continue their momentum and "
        "weak ones may correct further. By investing in the top-performing stocks, we aim to capture "
        "this momentum effect."
    )

    def __init__(self, lookback_days: int = 60, top_n_stocks: int = 10) -> None:
        self._lookback_days = lookback_days
        self._top_n_stocks = top_n_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days).sort("session_date").tail(self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the relative strength
        rel_strength_df = (
            history
            .select(pl.col("adj_close"))
            .transpose()
            .with_columns(
                (pl.lit(history["session_date"].max()) - pl.col("col")).alias("date_diff")
            )
            .rename({0: "symbol"})
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").last() / pl.col("adj_close").first()).alias("rs"),
                pl.min("date_diff").alias("latest_date_diff"),
            )
        )

        # Find the top N stocks based on relative strength
        rel_strength_df = rel_strength_df.sort("rs", descending=True).head(self._top_n_stocks)

        if rel_strength_df.height < self._top_n_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._top_n_stocks
        signal_weights = {row["symbol"]: weight for row in rel_strength_df.iter_rows()}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest