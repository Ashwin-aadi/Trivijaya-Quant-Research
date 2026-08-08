from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionDispersion(Strategy):
    rationale = (
        "This strategy exploits the market's tendency to oscillate between periods of high volatility (dispersion) "
        "and low volatility (range compression). During range compression phases, stocks tend to trade within a narrow band. "
        "During dispersion phases, markets exhibit greater price fluctuations. By identifying these phases and buying/selling "
        "at appropriate levels, we aim to profit from the reversion of market psychology."
    )

    def __init__(self, window_vol: int = 20, threshold_range_low: float = 0.15, threshold_range_high: float = 0.45,
                 lookback_support_resistance: int = 30) -> None:
        self._window_vol = window_vol
        self._threshold_range_low = threshold_range_low
        self._threshold_range_high = threshold_range_high
        self._lookback_support_resistance = lookback_support_resistance

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_vol * 2 + self._lookback_support_resistance)
        if history.height < self._window_vol * 2 + self._lookback_support_resistance:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns and rolling volatility
        close_history = history["adj_close"]
        daily_returns = (close_history / close_history.shift(1) - 1).alias("daily_return")
        rolling_volatility = close_history.rolling_window(
            window_size=self._window_vol,
            closed="both"
        ).std().alias("volatility")

        # Identify range compression and dispersion phases
        is_range_compression = (rolling_volatility < self._threshold_range_low).alias("is_range_compression")
        is_dispersion = (rolling_volatility > self._threshold_range_high).alias("is_dispersion")

        history_with_vols = history.with_columns(daily_returns, rolling_volatility, is_range_compression, is_dispersion)

        # Calculate support and resistance levels
        low_prices = history_with_vols["adj_close"].rolling_window(
            window_size=self._lookback_support_resistance,
            closed="both"
        ).min().alias("low_price")
        high_prices = history_with_vols["adj_close"].rolling_window(
            window_size=self._lookback_support_resistance,
            closed="both"
        ).max().alias("high_price")

        # Filter symbols based on current phase
        range_compression_symbols = (
            history_with_vols.filter(is_range_compression)
                             .select(["symbol", "session_date", "low_price"])
                             .sort("session_date", descending=True)
                             .head(1)
        )
        dispersion_symbols = (
            history_with_vols.filter(is_dispersion)
                             .select(["symbol", "session_date", "high_price"])
                             .sort("session_date", descending=True)
                             .head(1)
        )

        # Rank and select top symbols for each phase
        range_compression_picks: list[str] = []
        dispersion_picks: list[str] = []

        if not range_compression_symbols.is_empty():
            support_level = float(range_compression_symbols["low_price"][0])
            symbol = range_compression_symbols["symbol"][0]
            range_compression_picks.append(symbol)

        if not dispersion_symbols.is_empty():
            resistance_level = float(dispersion_symbols["high_price"][0])
            symbol = dispersion_symbols["symbol"][0]
            dispersion_picks.append(symbol)

        # Combine picks and assign weights
        picks = range_compression_picks + dispersion_picks
        weight = 1.0 / len(picks) if picks else 0

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