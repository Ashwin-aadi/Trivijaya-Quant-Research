from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that financial markets will tend to move towards the mean "
        "of their price series over time. In a short horizon, extreme prices are likely to revert "
        "to more typical levels. Identifying such extremes in the NIFTY 100 constituents can lead "
        "to profitable trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().select(
            pl.col(view.symbols).mean().alias("mean")
        ).get_column("mean").to_list()[0]
        std_deviation = (
            closes.select(pl.col(view.symbols).stddev()).to_numpy().item()
        )
        
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            z_score = (latest_close - mean_close) / std_deviation
            if z_score < -1.0 or z_score > 1.0:  # Identifying extreme values
                picks.append(symbol)

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