from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower risks and can provide more stable returns. "
        "By tilting our portfolio towards these stocks, we aim to reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean and standard deviation of returns over the lookback period
        returns = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .group_by("symbol")
            .agg(pl.col("r").mean().alias("mean_return"), pl.col("r").std().alias("std_return"))
        )

        # Sort by standard deviation of returns (lower is better for low-volatility selection)
        sorted_returns = returns.sort("std_return", descending=False)

        # Select the top N symbols based on lowest volatility
        top_symbols = [row["symbol"] for _, row in sorted_returns.rows()][:self._top_n]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest