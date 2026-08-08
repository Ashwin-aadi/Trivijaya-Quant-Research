from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Short-horizon mean reversion seeks to profit from temporary deviations of asset prices "
        "from their historical averages. By comparing each day's closing price to a 5-day moving "
        "average, this strategy aims to identify overbought or oversold conditions that are likely "
        "to revert."
    )

    def __init__(self, window: int = 5, max_loss: float = -0.05) -> None:
        self._window = window
        self._max_loss = max_loss

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        latest_close = {s: float(v) for s, v in zip(history["symbol"], history["adj_close"].to_list())}
        mean_price = sum(latest_close.values()) / len(latest_close)
        mean_prices = [mean_price] * (self._window + 1)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in latest_close:
                continue
            price_diff = latest_close[symbol] - mean_prices[self._window]
            weight = 0.02 / len(view.symbols)  # Equal weights among 30 names

            signals[symbol] = weight if abs(price_diff) > self._max_loss else 0.0

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in signals.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest