from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion seeks to profit from mean-reverting assets. When an asset is far from "
        "its recent average price, it tends to revert back towards the mean over a short period."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(history.columns) < view.symbols[0] + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean")))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"),
                ((pl.col("adj_close") / pl.col("mean")) - 1.0).alias("relative_deviation"),
            )
        )

        filtered = mean_close.filter(
            (pl.col("deviation") > 2.0) & (pl.col("relative_deviation").is_nan())
        )
        if filtered.height == 0:
            return Signal(information_available_at=stamp, weights={})

        top_gains: list[str] = []
        for symbol in view.symbols:
            row = mean_close.filter(pl.col("symbol") == symbol).row(0)
            deviation = float(row[1])
            if deviation > 2.0 and not row[2].is_nan():
                top_gains.append(symbol)

        weight = 1.0 / len(top_gains)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_gains}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest