from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy identifies significant price movements supported by substantial trading "
        "volume to gauge strong market sentiment and potential trend continuation. It aims to enter "
        "long positions only during directional moves that are volume-confirmed, reducing false signals."
    )

    def __init__(self, window: int = 5, lookback_days: int = 30, max_positions: int = 20) -> None:
        self._window = window
        self._lookback_days = lookback_days
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.height < 2 * (self._window + 1):
            return Signal(information_available_at=stamp, weights={})

        filtered_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            data = history.filter(pl.col("symbol") == symbol).sort("session_date")
            open_prices = [float(v) for v in data["open"].to_list()]
            high_prices = [float(v) for v in data["high"].to_list()]
            low_prices = [float(v) for v in data["low"].to_list()]
            close_prices = [float(v) for v in data["close"].to_list()]
            volume = [float(v) for v in data["volume"].to_list()]

            if len(open_prices) < self._lookback_days:
                continue

            # Check for directional move
            last_close = close_prices[-1]
            last_high = high_prices[-1]
            last_low = low_prices[-1]

            if (last_close > last_high and max(close_prices[:-1]) <= last_low) or \
               (last_close < last_low and min(close_prices[:-1]) >= last_high):
                # Check for volume confirmation
                average_volume = sum(volume[-self._window:]) / self._window
                if volume[-1] > 1.5 * average_volume:
                    filtered_symbols.append(symbol)

        filtered_symbols = filtered_symbols[:self._max_positions]
        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest