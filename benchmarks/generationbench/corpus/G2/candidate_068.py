from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can signal strong momentum and continuation of "
        "price trends. By identifying instances where a stock's volume significantly increases "
        "on the day following an up or down move, we can generate potentially profitable trades."
    )

    def __init__(self, window_up: int = 10, window_down: int = 10) -> None:
        self._window_up = window_up
        self._window_down = window_down

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_up + self._window_down)

        if history.height < self._window_up + self._window_down:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            hist = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            last_close = hist["close"].last()
            volume_yesterday = hist.select(pl.col("volume").last())
            volume_window_up = hist.sort(by="session_date", descending=False).head(self._window_up + 1)
            volume_window_down = hist.sort(by="session_date", descending=True).head(self._window_down + 1)

            if last_close.is_nan():
                continue

            # Calculate up and down moves
            up_move = (last_close - hist["close"].shift(1)).alias("up_move")
            down_move = ((hist["close"].shift(1) - last_close).abs()).alias("down_move")

            # Check for volume increase after an up move or down move
            if (up_move.last() > 0 and
                volume_yesterday[0] > volume_window_up.select(pl.col("volume").mean().alias("avg_volume")).item()):
                signals[symbol] = 1.0 / len(signals) + 1e-6

            elif (down_move.last() > 0 and
                  volume_yesterday[0] > volume_window_down.select(pl.col("volume").mean().alias("avg_volume")).item()):
                signals[symbol] = -1.0 / len(signals) - 1e-6

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest