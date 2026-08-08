from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum and can lead to "
        "significant gains. By identifying such moves early, we can capitalize on the trend."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_price = [float(v) for v in history[symbol]["open"].to_list()]
            close_price = [float(v) for v in history[symbol]["close"].to_list()]
            volume = [float(v) for v in history[symbol]["volume"].to_list()]

            if len(open_price) < self._window or len(close_price) < self._window or len(volume) < self._window:
                continue

            # Calculate daily returns
            returns = [(close / open - 1.0) for close, open in zip(close_price[1:], open_price[:-1])]
            volume_shifted = [v / v_prev if v != 0 and v_prev != 0 else 0 for v, v_prev in zip(volume[1:], volume[:-1])]

            # Find directional moves with significant volume
            recent_return = returns[-2]
            recent_volume = volume[-2] / volume[-3] if len(volume) > 2 else 0

            if recent_return >= 0.05 and recent_volume >= 1.5:
                signals.append(symbol)
            elif recent_return <= -0.05 and recent_volume >= 1.5:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest