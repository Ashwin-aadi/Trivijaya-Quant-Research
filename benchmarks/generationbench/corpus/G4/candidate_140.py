from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "The strategy aims to capitalize on volume-confirmed directional moves in the Indian equity market. "
        "High trading volumes often precede significant price movements due to increased market participation and liquidity."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        signals = {}
        for symbol in view.symbols:
            close_series = history.select(pl.col(symbol).alias("close"))
            volume_series = history.select(pl.col(symbol).alias("volume"))

            close = float(close_series["close"][0])
            prev_high = float(close_series["close"].max())
            prev_low = float(close_series["close"].min())
            curr_volume = float(volume_series["volume"][-1])
            prev_ma20_volume = float(volume_series["volume"].mean())

            if (close > prev_high) and (curr_volume > prev_ma20_volume):
                signals[symbol] = "buy"

            if (close < prev_low) and (curr_volume < prev_ma20_volume):
                signals[symbol] = "sell"

        buys = {s: 1.0 for s in signals if signals[s] == "buy"}
        sells = {s: -1.0 for s in signals if signals[s] == "sell"}

        combined_weights = {**buys, **sells}
        if not combined_weights:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=combined_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest