from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy aims to exploit the anomaly in small-cap stocks by equal-weighting "
        "liquid candidates. By focusing on liquidity and equal weighting, we aim to manage risk "
        "while capturing potential returns from mispriced expectations or market inefficiencies."
    )

    def __init__(self, window: int = 30, min_volume: float = 10_000, lookback: int = 60) -> None:
        self._window = window
        self._min_volume = min_volume
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        filtered_history = history.filter(pl.col("symbol").is_in(symbols))

        # Filter by liquidity
        liquidity_filter = (
            (filtered_history.select(pl.col("volume").mean().over("symbol")) > self._min_volume)
            & (filtered_history.select(pl.col("bidaskspread").mean().over("symbol")) < 0.1)
        )
        liquid_symbols = filtered_history.filter(liquidity_filter).select("symbol").to_series().to_list()

        if not liquid_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting
        weight = 1.0 / len(liquid_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquid_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest