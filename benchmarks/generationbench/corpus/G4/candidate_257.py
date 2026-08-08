from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy capitalizes on breakout continuation by identifying stocks that have "
        "broken out of a consolidation pattern and then continue to move in the breakout direction. "
        "The idea is that such breakouts often indicate a change in trend, leading to sustained price movements."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)
        if history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        breakout_highs = _find_breakout_highs(closes)
        breakout_lows = _find_breakout_lows(closes)

        long_signals = []
        short_signals = []

        for symbol in view.symbols:
            if (
                symbol not in history.columns
                or symbol not in closes.columns
                or symbol not in breakout_highs.columns
                or symbol not in breakout_lows.columns
            ):
                continue

            # Calculate breakout high and low levels
            breakout_high = float(breakout_highs[symbol].max())
            breakout_low = float(breakout_lows[symbol].min())

            # Get latest close price and volume data
            latest_close = view.latest_close()[symbol]
            volume_20d_avg = history.select(
                pl.col("symbol").eq(symbol).alias("match"),
                pl.col("volume").mean().over(pl.arange()).alias("avg_volume"),
            ).filter(pl.col("match"))["avg_volume"].item()

            # Calculate breakout continuation for long and short
            if latest_close > breakout_high:
                pct_increase = (latest_close - breakout_high) / breakout_high
                if _is_strong_signal(volume_20d_avg, history[symbol]["volume"][-1]):
                    long_signals.append((symbol, pct_increase))
            elif latest_close < breakout_low:
                pct_decrease = (breakout_low - latest_close) / breakout_low
                if _is_strong_signal(volume_20d_avg, history[symbol]["volume"][-1]):
                    short_signals.append((symbol, pct_decrease))

        long_signals.sort(key=lambda x: x[1], reverse=True)
        short_signals.sort(key=lambda x: x[1], reverse=False)

        top_long_symbols = [s[0] for s in long_signals[: self._top_n]]
        top_short_symbols = [s[0] for s in short_signals[: self._top_n]]

        if not top_long_symbols and not top_short_symbols:
            return Signal(information_available_at=stamp, weights={})

        long_weight = 1.0 / max(len(top_long_symbols), 1)
        short_weight = -1.0 / max(len(top_short_symbols), 1)

        weights = {s: weight for s, weight in [(symbol, long_weight) for symbol in top_long_symbols] + [(symbol, short_weight) for symbol in top_short_symbols]}
        return Signal(information_available_at=stamp, weights=weights)


def _find_breakout_highs(closes: pl.DataFrame) -> pl.DataFrame:
    return closes.sort("session_date").group_by("symbol").agg(
        (pl.col("adj_close") - pl.col("adj_close").shift(1)).cummax().alias("high_diff"),
        pl.col("adj_close").where(pl.col("high_diff") == 0).first().alias("breakout_high"),
    )


def _find_breakout_lows(closes: pl.DataFrame) -> pl.DataFrame:
    return closes.sort("session_date").group_by("symbol").agg(
        (pl.col("adj_close") - pl.col("adj_close").shift(1)).cummin().alias("low_diff"),
        pl.col("adj_close").where(pl.col("low_diff") == 0).first().alias("breakout_low"),
    )


def _is_strong_signal(volume_20d_avg: float, recent_volume: int) -> bool:
    return volume_20d_avg * 0.8 < recent_volume


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest