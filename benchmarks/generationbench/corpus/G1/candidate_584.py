from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "A significant volume increase alongside a price breakout suggests strong buying or selling "
        "pressure. This can indicate the start of a trend and provide an opportunity for profitable trades."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            volumes = [float(v) for v in history[f"{symbol}_volume"].to_list()]

            if len(prices) < self._window + 1 or len(volumes) < self._window + 1:
                continue

            # Calculate price change and volume
            recent_close = prices[-1]
            previous_close = prices[-2]

            # Check for breakout condition
            breakout_condition = (recent_close - previous_close) / previous_close > 0.01

            if not breakout_condition:
                continue

            # Check for volume increase
            recent_volume = volumes[-1]
            previous_volume = volumes[-2]
            volume_increase_condition = recent_volume / previous_volume >= self._threshold

            if volume_increase_condition:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]  # Limit to top 5 symbols
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest