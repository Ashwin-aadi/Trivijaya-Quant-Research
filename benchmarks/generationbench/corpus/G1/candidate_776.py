from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate a stronger trend in the market. "
        "When there is an increase in volume alongside a price move, it often suggests "
        "greater investor conviction and potential continuation of the trend."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_changes = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            symbol_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            )
            opens = [float(v) for v in symbol_history["open"].to_list()]
            closes = [float(v) for v in symbol_history["close"].to_list()]
            volumes = [int(v) for v in symbol_history["volume"].to_list()]

            # Calculate price change
            price_changes = [
                (c - o) / o if o != 0 else float("nan") for o, c in zip(opens[:-1], closes[1:])
            ]
            price_change = max(price_changes, default=0)

            # Calculate volume increase
            last_volume = volumes[-1]
            prev_volume = volumes[-2] if len(volumes) > 1 else 0
            volume_increase = (last_volume - prev_volume) / prev_volume if prev_volume != 0 else float("nan")

            # Check for volume-confirmed move
            if price_change >= 0.05 and volume_increase >= 0.2:
                volume_changes[symbol] = price_change

        if not volume_changes:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = max(volume_changes, key=lambda k: volume_changes[k])
        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest