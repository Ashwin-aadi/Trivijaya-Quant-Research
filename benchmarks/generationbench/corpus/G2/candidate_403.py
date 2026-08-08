from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when stock prices are trading within a narrow range relative "
        "to their recent volatility. This suggests that the market is consolidating and may be "
        "due to break out in the near future, providing an opportunity for profitable trades."
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
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            high_price = max(prices)
            low_price = min(prices)

            # Calculate the range compression ratio
            range_ratio = (high_price - low_price) / sum(prices)

            # If the range is compressed, add to picks
            if range_ratio <= 0.1:
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest