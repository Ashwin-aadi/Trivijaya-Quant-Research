from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is because low-volatility stocks are less risky and can provide more stable returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean and standard deviation of returns for each stock
        history = (
            history.group_by("symbol")
                   .agg(
                       (pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1).alias("return"),
                   )
                   .sort("session_date", descending=False)
                   .with_columns(
                       (pl.col("return").mean().over(pl.col("symbol"))).alias("mean_return"),
                       (pl.col("return").stddev().over(pl.col("symbol"))).alias("volatility"),
                   )
        )

        # Select the low-volatility symbols
        sorted_history = history.sort("volatility", descending=False)
        picks: list[str] = [row["symbol"] for row in sorted_history.to_dicts()][:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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