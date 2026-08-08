from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendVolatility(Strategy):
    rationale = (
        "By combining a 5-day moving average of daily closing prices with the 7-day volatility (standard deviation of daily returns), we aim to capture both the price trend and market sentiment. A strong upward trend combined with low volatility suggests a robust bullish momentum, while a weak trend or high volatility may indicate caution."
    )

    def __init__(self, ma_window: int = 5, vol_window: int = 7) -> None:
        self._ma_window = ma_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._ma_window + self._vol_window - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("adj_close").to_list())

        ma_values = [float(v) for v in (
            (closes["adj_close"].shift(-i).rolling_mean(self._ma_window))
        ).drop_nulls().to_list()]

        returns = [(close - open_) / open_ for open_, close in zip(
            closes["adj_close"].shift(1).drop_nulls().to_list(),
            closes["adj_close"].drop_nulls().to_list()
        )]

        vol_values = [float(v) for v in (
            (pl.Series(returns).rolling_std(self._vol_window))
        ).drop_nulls().to_list()]

        if len(ma_values) < self._ma_window or len(vol_values) < self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            ma_last = ma_values[-1][symbol]
            vol_last = vol_values[-1][symbol]

            if ma_last >= max(ma_values) and vol_last <= min(vol_values):
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest