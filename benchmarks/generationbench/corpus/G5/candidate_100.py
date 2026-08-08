from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAvg(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying assets whose prices have moved far "
        "away from their trailing average, we can identify potential mean-reverting opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("avg_close")))
            .with_columns(
                (pl.col("close") / pl.col("avg_close").over("symbol") - 1.0).alias("reversion_factor")
            )
        )

        symbols = [s for s in view.symbols if not avg_close.is_empty()]
        filtered_history = history.select(["session_date", "symbol"] + symbols)
        filtered_avg_close = avg_close.filter(pl.col("avg_close").is_not_null())

        reversion_factors = (
            filtered_history.join(filtered_avg_close, on="symbol")
            .with_columns(
                (pl.col("close") / pl.col("avg_close") - 1.0).alias("reversion_factor")
            )
            .select("session_date", "symbol", "reversion_factor")
        )

        reversion_factors = (
            reversion_factors.filter(
                (pl.col("reversion_factor").abs() >= self._threshold)
            ).sort("reversion_factor", descending=True)
            .to_pandas()
        )

        symbols = reversion_factors.iloc[:5, 1].tolist()  # Select top 5 most reverted
        weight = 1.0 / len(symbols) if symbols else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest