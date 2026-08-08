from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-screened equal weighting aims to distribute capital more efficiently by "
        "considering the liquidity of assets. Highly liquid stocks are favored, ensuring that "
        "trades can be executed without significantly impacting the stock price."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume"),
                (pl.col("adj_close") - pl.col("open")).abs()
                / pl.col("open")
                .shift(1)
                .is_not_null()  # Remove NaNs for first day
                .cast(pl.Float64)  # Ensure the division is float
                .alias("volatility"),
            )
            .select(["symbol", "avg_volume", "volatility"])
        )

        if liquidity_scores.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = (
            liquidity_scores.sort(
                ["avg_volume", "volatility"], descending=[True, True]
            )
            .select("symbol")
            .to_series()
            .to_list()[: self._window]
        )

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest