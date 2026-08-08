from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate a strong commitment to the direction of "
        "the price. Such moves often lead to sustained momentum and can be used to identify "
        "opportunities in the market."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 1e6) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        moves: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            volumes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["volume"].to_list()]

            if len(prices) < self._window:
                continue

            for i in range(1, len(prices)):
                direction = (prices[i] - prices[i-1]) / abs(prices[i-1])
                volume_ratio = volumes[i] / volumes[i-1]

                if abs(direction) > 0.05 and volume_ratio >= 2:
                    moves[symbol] = direction * volume_ratio

        sorted_moves = sorted(moves.items(), key=lambda x: abs(x[1]), reverse=True)
        picks = [symbol for symbol, _ in sorted_moves[:3]]  # Select top 3 strongest moves
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest