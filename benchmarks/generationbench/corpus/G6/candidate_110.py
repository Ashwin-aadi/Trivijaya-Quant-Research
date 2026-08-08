from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy combines Moving Average crossovers and MACD indicators to identify trends with enhanced reliability. "
        "It aims to enter positions when the short-term MA crosses above the long-term MA or both the MACD line crosses above the signal line, "
        "and exits based on a combination of EMA deviation and holding period limits."
    )

    def __init__(self, window_short: int = 50, window_long: int = 200, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> None:
        self._window_short = window_short
        self._window_long = window_long
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_long + 10)

        if history.height < self._window_short + 5:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            df = history.select(
                pl.col("symbol"), pl.col("session_date"), 
                pl.col("close").alias("c"), pl.col("adj_close").alias("a")
            ).filter(pl.col("symbol") == symbol).sort("session_date")

            if df.height < self._window_short + 5:
                continue

            ma_short = (df["c"].rolling_window(self._window_short, closed="right").mean()).shift(-1)
            ma_long = (df["c"].rolling_window(self._window_long, closed="right").mean()).shift(-1)

            macd_line = (df["a"].ewm(alpha=1/self._macd_fast).mean() - df["a"].ewm(alpha=1/self._macd_slow).mean())
            signal_line = macd_line.rolling_window(self._macd_signal, closed="right").mean()

            if ma_short[0] > ma_long[0]:
                signals[symbol] = 1.0

            trend_crossed = (ma_short.shift(-2) < ma_long.shift(-2)) & (ma_short[-1] > ma_long[-1])
            macd_crossed = (macd_line.shift(-2) < signal_line.shift(-2)) & (macd_line[-1] > signal_line[-1])

            if trend_crossed:
                signals[symbol] = 1.0
            elif macd_crossed:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest