from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout20d(Strategy):
    rationale = (
        "A strong close in the top of its recent volume range suggests buying pressure. "
        "Combining this with a price breakout above its recent high can indicate a bullish signal."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_closes = view.closes().select("symbol", "session_date", pl.col("volume").cumsum())
        volume_breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in volume_closes["symbol"].to_list():
                continue
            price_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            volume_values = [int(v) for v in volume_closes.filter(pl.col("symbol") == symbol)["volume"].to_list()]
            
            if len(price_values) < self._window or len(volume_values) < self._window:
                continue

            if price_values[-1] >= max(price_values):
                last_price = price_values[-1]
                top_volume_index = volume_values.index(max(volume_values))  # Find the day with maximum volume
                breakout_day_price = price_values[top_volume_index]

                if last_price > breakout_day_price:
                    volume_breakout_symbols.append(symbol)

        volume_breakout_symbols = volume_breakout_symbols[: self._top_n]
        if not volume_breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volume_breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in volume_breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest