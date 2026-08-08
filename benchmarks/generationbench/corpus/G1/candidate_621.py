from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over the long term. "
        "By tilting our portfolio towards low-volatility equities, we aim to capture this effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Group by symbol and calculate the standard deviation of returns over the window period
        std_devs = (
            history.group_by("symbol")
            .agg(
                (pl.col("return").std().alias("volatility"))
            )
            .sort("volatility", descending=False)
            .select(["symbol", "volatility"])
        )

        # Convert to list of tuples for easier manipulation
        volatility_list = std_devs.to_dict(as_series=False).values()

        # Select the top N symbols based on lowest volatility
        picks: list[str] = [vol[0] for vol in volatility_list[:5]]

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