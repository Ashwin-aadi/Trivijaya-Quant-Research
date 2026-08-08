from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to continue. By identifying symbols "
        "that have recently made a significant move in one direction and confirmed it with "
        "volume, we can capitalize on the momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Identify symbols with significant directional moves
        directional_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue

            prices = [float(v) for v in history[history["symbol"] == symbol]["adj_close"].to_list()]
            direction_changes = [
                (prices[i] - prices[i-1]) / abs(prices[i-1])
                for i in range(1, len(prices))
            ]
            if all(d > self._threshold for d in direction_changes):
                directional_symbols.append(symbol)
            elif all(d < -self._threshold for d in direction_changes):
                directional_symbols.append(symbol)

        # Filter symbols with volume confirmation
        volume_confirmed = []
        for symbol in directional_symbols:
            vol_history = view.history(lookback=self._window).filter(
                pl.col("symbol") == symbol
            )
            volumes = [float(v) for v in vol_history["volume"].to_list()]
            if any(volumes[i] > 1.2 * max(volumes[:i]) for i in range(self._window, len(volumes))):
                volume_confirmed.append(symbol)

        # Select top N symbols
        volume_confirmed = volume_confirmed[:5]
        if not volume_confirmed:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(volume_confirmed)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in volume_confirmed}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest