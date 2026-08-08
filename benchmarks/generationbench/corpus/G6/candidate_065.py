from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Identify stocks exhibiting strong upward breakouts from recent trading ranges using "
        "daily OHLCV data. Confirm breakouts with high volume and RSI > 70 to ensure momentum."
    )

    def __init__(self, window: int = 50, threshold: float = 1.03, rsi_threshold: int = 70, hold_days: int = 15) -> None:
        self._window = window
        self._threshold = threshold
        self._rsi_threshold = rsi_threshold
        self._hold_days = hold_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 10).sort("session_date")
        if history.height < self._window + 10:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        breakout_symbols: list[str] = []
        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol)
            close_series = df.select("close").to_pandas().squeeze()
            volume_series = df.select("volume").to_pandas().squeeze()
            rsi = _compute_rsi(close_series, self._window)

            if any(rsi < self._rsi_threshold):
                continue

            breakout_price = close_series[-1]
            last_close = close_series[-2]

            if (breakout_price - last_close) / last_close >= self._threshold and volume_series[-1] > volume_series.rolling(window=21).mean()[-1]:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(price_series: pl.Series, window: int) -> pl.Series:
    delta = price_series.diff().drop_nulls()
    up, down = delta.clip(lower=0), -delta
    ema_up = up.rolling_mean(window_size=window).alias("ema_up")
    ema_down = down.rolling_mean(window_size=window).alias("ema_down")
    rsi = (100 * ema_up / (ema_up + ema_down)).round().cast(pl.Int32)
    return rsi