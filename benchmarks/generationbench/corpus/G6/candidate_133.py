from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Leverage short-term mean reversion by identifying stocks with daily closing prices "
        "deviating more than 3 standard deviations from a 20-day simple moving average (SMA). "
        "This strategy balances statistical rigor with practical risk management."
    )

    def __init__(self, window: int = 20, std_dev_lookback: int = 5, max_positions: int = 30) -> None:
        self._window = window
        self._std_dev_lookback = std_dev_lookback
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < (self._window + self._std_dev_lookback):
            return Signal(information_available_at=stamp, weights={})

        sma_20 = (
            closes
            .group_by("symbol")
            .agg((pl.col("adj_close").sum() / pl.col("adj_close").count()).alias("sma_20"))
        )
        std_dev_5 = (
            closes
            .group_by("symbol")
            .agg(pl.col("adj_close").std().over(window_size=self._std_dev_lookback).alias("std_dev_5"))
        )

        merged = sma_20.join(std_dev_5, on="symbol", how="inner")

        for symbol in view.symbols:
            if symbol not in merged.columns or len(merged[symbol].to_list()) < self._window + 1:
                continue
            values = [float(v) for v in merged[symbol].drop_nulls().to_list()]
            z_score = (values[-1] - values[:self._window].mean()) / values[self._window:].std()
            if z_score <= -3.0:
                picks = symbol

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._max_positions
        return Signal(
            information_available_at=stamp, weights={picks: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest