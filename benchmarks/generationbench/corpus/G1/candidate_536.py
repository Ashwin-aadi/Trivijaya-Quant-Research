from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can signal "
        "a potential trend continuation or acceleration. By identifying these moves early, we "
        "can capture significant gains before the trend reverses."
    )

    def __init__(self, window: int = 20, min_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._min_volume_ratio = min_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        moves: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            data = history.select(["session_date", "symbol", "open", "close", "volume"])
            open_prices = [float(v) for v in data["open"].to_list()]
            close_prices = [float(v) for v in data["close"].to_list()]
            volumes = [float(v) for v in data["volume"].to_list()]

            if len(open_prices) < self._window or len(close_prices) < self._window:
                continue

            for i in range(self._window - 1):
                if (open_prices[i] <= close_prices[i]) and (
                    volumes[i + 1] / volumes[i] >= self._min_volume_ratio
                ):
                    moves.append(symbol)
                    break

        moves = list(set(moves))[: self._window]
        if not moves:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(moves)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest