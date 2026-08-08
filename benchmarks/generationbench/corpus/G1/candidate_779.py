from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines the daily return volatility with the 20-day moving average "
        "to identify stocks that are both less volatile and above their moving average. The "
        "idea is to find stocks that are stable but still have a positive trend, potentially "
        "indicating strong fundamental health."
    )

    def __init__(self, vol_window: int = 20, ma_window: int = 20) -> None:
        self._vol_window = vol_window
        self._ma_window = ma_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._vol_window + self._ma_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_vols = {}
        ma_crosses = {}

        for symbol in view.symbols:
            if symbol not in history.columns or "session_date" not in history.columns:
                continue

            adj_closes = history[symbol].to_list()
            if len(adj_closes) < self._vol_window + self._ma_window - 1:
                continue

            # Calculate daily returns
            returns = [float(v) for v in (pl.Series(adj_closes).diff().drop_nulls()).to_list()]
            
            # Volatility over the lookback period
            vol = pl.Series(returns).std()

            # Moving average cross signal
            ma_series = pl.Series(adj_closes[-self._ma_window:]).rolling_mean(window_size=self._ma_window)
            last_ma = float(ma_series[-1])
            prev_ma = float(ma_series[-2])
            if last_ma > prev_ma and returns[-1] >= 0:
                ma_crosses[symbol] = True
            else:
                ma_crosses[symbol] = False

            symbol_vols[symbol] = vol

        # Identify symbols that are above their moving average and have low volatility
        picks = [s for s, v in symbol_vols.items() if ma_crosses.get(s) and v < 0.1]

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