from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout(Strategy):
    rationale = (
        "Volume breakout strategies exploit the idea that a significant increase in volume "
        "accompanying a price move can indicate strong underlying demand or supply. By focusing "
        "on symbols where both price and volume show a directional move, we aim to identify "
        "sustainable trends."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            prices = [float(v) for v in history[symbol, "close"].drop_nulls().to_list()]
            volumes = [float(v) for v in history[symbol, "volume"].drop_nulls().to_list()]

            # Calculate the price and volume change from the previous day
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0.0 for i in range(len(prices))]
            volume_changes = [volumes[i] - volumes[i-1] if i > 0 else 0.0 for i in range(len(volumes))]

            # Check for a significant price and volume move
            last_price_change = price_changes[-1]
            last_volume_change = volume_changes[-1]
            if (last_price_change > 0 and max(price_changes) == last_price_change and
                last_volume_change > sum(volume_changes[:-1]) / len(volume_changes)):
                breakout_symbols.append(symbol)

        weights = {symbol: 1.0 / len(breakout_symbols) for symbol in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest