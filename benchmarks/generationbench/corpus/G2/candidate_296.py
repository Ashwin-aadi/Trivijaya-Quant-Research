from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are expected to be more robust and less noisy than "
        "price movements alone. Higher trading volume often signals a stronger conviction among "
        "market participants, which can lead to sustained price trends."
    )

    def __init__(self, window: int = 20, volume_multiplier: float = 1.5) -> None:
        self._window = window
        self._volume_multiplier = volume_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter symbols with sufficient trading data
        symbols_with_data = [s for s in view.symbols if all(s in h.columns for h in (view.closes(), view.history()))]
        
        signals: dict[str, float] = {}
        for symbol in symbols_with_data:
            history_df = history.select(["session_date", f"{symbol}_open", f"{symbol}_close", f"{symbol}_volume"])
            if history_df.height < self._window:
                continue

            # Calculate daily returns and volume
            price_changes = (history_df[f"{symbol}_close"] / history_df[f"{symbol}_open"].shift(1) - 1.0).to_list()[1:]
            volumes = history_df[f"{symbol}_volume"].to_list()[1:]

            if all(v <= self._volume_multiplier * v_24_before for v, v_24_before in zip(volumes, [volumes[0]] + volumes[:-1])):
                # Check for a volume-confirmed directional move
                if (price_changes[-1] > 0 and price_changes[:self._window].count(0) == self._window - 1) or \
                   (price_changes[-1] < 0 and price_changes[:self._window].count(0) == self._window - 1):
                    signals[symbol] = 1.0 / len(symbols_with_data)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        
        return Signal(
            information_available_at=stamp,
            weights=signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest