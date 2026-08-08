from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate a strong trend. If a stock's price "
        "moves in the same direction as its volume, it suggests that institutional or large "
        "traders are driving the price action, potentially leading to continuation of the move."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)

            price_move = (
                (symbol_history["close"] / symbol_history["close"].shift(1) - 1.0).mean()
            ).to_numpy()[0]
            volume_move = (
                (symbol_history["volume"] / symbol_history["volume"].shift(1)).mean()
            ).to_numpy()[0]

            if price_move > 0 and volume_move > 1.0:
                picks.append(symbol)

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