from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout in one direction, the price may continue to move in that same "
        "direction. Identifying such continuations can provide trading opportunities."
    )

    def __init__(self, window: int = 20, min_price_change: float = 0.05) -> None:
        self._window = window
        self._min_price_change = min_price_change

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window + 1:
                continue

            breakout_price = max(prices[: self._window])
            current_price = prices[self._window]
            price_change = (current_price - breakout_price) / breakout_price
            if price_change >= self._min_price_change:
                picks.append(symbol)

        picks = picks[:5]  # Select top 5 symbols for simplicity
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
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