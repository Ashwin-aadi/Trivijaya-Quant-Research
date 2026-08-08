from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that after an asset deviates significantly from its mean "
        "price over a period, it tends to revert back. We look for stocks that have had large "
        "downside moves in the last 10 days and expect them to bounce back."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close").mean().over("symbol")).alias("10d_mean")
        ).collect()

        below_mean = (
            history
            .with_columns((pl.col("adj_close") - pl.col("10d_mean")).alias("deviation"))
            .filter(pl.col("deviation") < 0)
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").min().over("symbol")).alias("min_price"),
                (pl.col("deviation").abs().max().over("symbol")).alias("max_deviation")
            )
            .sort("max_deviation", descending=True)
        ).collect()

        picks: list[str] = []
        for symbol, row in below_mean.iter_rows():
            if mean_close.get_column("10d_mean").to_list()[mean_close.get_column("symbol").to_list().index(symbol)] - row["min_price"] >= 2 * row["max_deviation"]:
                picks.append(symbol)

        picks = [p for p in picks if view.symbols.count(p) > 0]
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