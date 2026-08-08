from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy aims to capitalize on large-volume trades that confirm a clear directional move in stock prices. "
        "High trading volume often precedes significant price movements due to increased market participation and information dissemination. "
        "High volumes can indicate strong buying or selling pressure, making these moves more reliable."
    )

    def __init__(self, volume_window: int = 20, sma_short: int = 50, sma_long: int = 200, rsi_window: int = 14, top_n: int = 7) -> None:
        self._volume_window = volume_window
        self._sma_short = sma_short
        self._sma_long = sma_long
        self._rsi_window = rsi_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._volume_window + 100)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        volume = history["volume"].to_list()
        symbols = [str(symbol) for symbol in view.symbols]

        # Calculate daily volume deviation from 20-day moving average
        volume_deviation = []
        for i in range(self._volume_window, len(closes)):
            avg_volume = sum(volume[i - self._volume_window:i]) / self._volume_window
            daily_volume = volume[i]
            if avg_volume > 0:
                deviation = (daily_volume - avg_volume) / avg_volume
            else:
                deviation = 0.0
            volume_deviation.append(deviation)

        # Calculate 50-period SMA and 200-period SMA for trend confirmation
        sma_short_values = [float(v) for v in history.select(pl.col("adj_close").rolling_mean(self._sma_short)).to_dict().get("adj_close", [])]
        sma_long_values = [float(v) for v in history.select(pl.col("adj_close").rolling_mean(self._sma_long)).to_dict().get("adj_close", [])]

        # Calculate RSI
        rsi_values = [float(v) for v in view.closes(lookback=self._rsi_window).select(
            [(pl.col(s) / pl.col(s).shift(14) - 1.0).alias(f"r") for s in symbols]).to_dict().get("r", [])]

        # Rank stocks based on volume deviation, trend confirmation, and RSI
        scores = []
        for i, (symbol, vol_dev, sma_short, sma_long, rsi) in enumerate(zip(symbols, volume_deviation, sma_short_values[-10:], sma_long_values[-10:], rsi_values[-10:])):
            trend_signal = 1 if sma_short > sma_long else -1
            scores.append((symbol, vol_dev, trend_signal, rsi))

        # Filter and rank based on criteria
        ranked_scores = sorted(scores, key=lambda x: (abs(x[1]), abs(x[2]), -x[3]), reverse=True)[:self._top_n]

        if not ranked_scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_scores)
        selected_symbols = [score[0] for score in ranked_scores]
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest