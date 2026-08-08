from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price reversion strategies capitalize on the tendency of asset prices to return to a "
        "mean level over time. By using a trailing mean of historical prices as a reference, we "
        "can identify instances where the price deviates significantly from this mean and "
        "potentially signals an opportunity for reversion."
    )

    def __init__(self, window: int = 60, lookback: int = 30) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)

        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        trailing_mean = (
            closes.group_by("symbol")
            .agg((pl.col("adj_close").shift(-1).rolling_sum(self._window) / self._window).alias("trailing_mean"))
            .with_columns(pl.col("trailing_mean").fill_null(0.0))
        )

        history = view.history()
        symbols_with_trailing_mean = [symbol for symbol in trailing_mean.columns if symbol != "session_date"]
        
        signals: dict[str, float] = {}
        for symbol in symbols_with_trailing_mean:
            latest_close = float(view.latest_close()[symbol])
            trailing_mean_value = float(trailing_mean.filter(pl.col("symbol") == symbol).select("trailing_mean").to_dict(False)[0][1])
            deviation = (latest_close - trailing_mean_value) / trailing_mean_value

            if abs(deviation) > 0.2:
                signals[symbol] = 1.0 / len(signals)

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest