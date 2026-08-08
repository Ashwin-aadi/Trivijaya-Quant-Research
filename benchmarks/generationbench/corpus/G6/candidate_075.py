from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy selects stocks based on sufficient liquidity to minimize market impact and "
        "risks associated with illiquid securities. It ensures an equal weight for each selected stock, "
        "providing a balanced exposure across the market."
    )

    def __init__(self, window: int = 30, min_volume: float = 500_000, num_stocks: int = 30) -> None:
        self._window = window
        self._min_volume = min_volume
        self._num_stocks = num_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_data = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume"),
            )
            .filter(pl.col("avg_volume") > self._min_volume)
            .select(["symbol"])
        )

        if volume_data.height < 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in volume_data["symbol"].to_list()]
        if len(symbols) < self._num_stocks:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = symbols[: self._num_stocks]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest