from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean after a strong move. By using a trailing reference, we can "
        "identify when prices have moved too far and are likely to correct back towards the "
        "mean."
    )

    def __init__(self, window: int = 50, mean_window: int = 20) -> None:
        self._window = window
        self._mean_window = mean_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_price = sum(prices[-self._mean_window:]) / self._mean_window

            last_price = prices[-1]
            change_ratio = (last_price - mean_price) / mean_price if mean_price != 0 else 0
            trend_strength = abs(change_ratio)

            # Check if the price move is significant compared to its recent range
            if trend_strength > 2 * (max(prices) - min(prices)) / 5:
                symbol_prices[symbol] = last_price

        picks = sorted(symbol_prices.keys(), key=lambda s: symbol_prices[s], reverse=True)
        weight = 1.0 / len(picks) if picks else 0
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