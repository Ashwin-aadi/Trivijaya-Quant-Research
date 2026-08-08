from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices which have deviated significantly from their "
        "historical mean will revert to it. In a short horizon, we can identify stocks that are "
        "trading far from their 20-day moving average and expect them to move back towards this "
        "average."
    )

    def __init__(self, lookback_window: int = 20) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window:
            return Signal(information_available_at=stamp, weights={})

        signal_strengths = {symbol: 0.0 for symbol in view.symbols}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_close_series = [float(v) for v in closes[symbol].to_list()]
            mean_adj_close = sum(adj_close_series[-self._lookback_window:]) / self._lookback_window
            latest_price = adj_close_series[-1]
            deviation = abs(latest_price - mean_adj_close)
            signal_strengths[symbol] = deviation

        sorted_signals = sorted(signal_strengths.items(), key=lambda x: x[1], reverse=True)
        if not sorted_signals:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = sorted_signals[0][0]
        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest