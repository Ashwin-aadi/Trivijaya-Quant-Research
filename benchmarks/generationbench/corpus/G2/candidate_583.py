from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price volatility decreases significantly. "
        "During such periods, the market tends to consolidate within a narrow range, often leading "
        "to increased trading opportunities as prices eventually break out of this range."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_range_compression = []

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            daily_highs = [float(v) for v in history.select(pl.col(symbol).max()).to_dict()[symbol]]
            daily_lows = [float(v) for v in history.select(pl.col(symbol).min()).to_dict()[symbol]]

            if len(daily_highs) != self._window or len(daily_lows) != self._window:
                continue

            max_range = max(high - low for high, low in zip(daily_highs, daily_lows))
            recent_max_range = max(daily_highs[-10:] + daily_lows[-10:])

            if max_range > 2 * recent_max_range and recent_max_range < 0.5:
                symbols_with_range_compression.append(symbol)

        weights = {symbol: 1.0 / len(symbols_with_range_compression) for symbol in symbols_with_range_compression}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest