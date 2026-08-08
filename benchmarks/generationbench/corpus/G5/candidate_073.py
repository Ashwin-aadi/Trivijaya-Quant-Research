from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that security prices and trading volumes that drift far from "
        "their historical average will eventually return. By identifying stocks that have deviated "
        "significantly from their 10-day mean, we can capitalize on the tendency for these prices to revert."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.groupby("symbol")
            .agg((pl.col("adj_close").mean()).alias("mean"))
            .with_columns(
                (pl.col("mean") - pl.col("close").shift(1).fill_none(0)).abs()
                .alias("diff")
            )
            .filter(pl.col("diff") > self._threshold)
        )

        if mean_close.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_mean_map: dict[str, float] = (
            mean_close.select(["symbol", "mean"])
            .with_columns(
                (pl.col("mean") - pl.col("close").shift(1).fill_none(0))
                / pl.col("adj_close")
                .std()
                .over("symbol")
                .alias("z_score")
            )
            .filter(pl.col("z_score").abs() > 2)
            .to_dict(as_series=False)  # type: ignore
        )

        if not symbol_mean_map:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbol_mean_map)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in symbol_mean_map.keys()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest