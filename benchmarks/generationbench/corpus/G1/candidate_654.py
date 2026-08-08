from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength outperform the market over the long term. "
        "By investing in the top-performing stocks relative to the broader universe, we aim to capture this trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window or closes.width == 1:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            avg_price = sum(prices[-10:]) / 10.0  # Simple moving average over the last 10 days
            strength = prices[-1] / avg_price - 1.0
            relative_strengths.append(strength)

        top_n_indices = sorted(range(len(relative_strengths)), key=lambda i: relative_strengths[i], reverse=True)[:5]
        picks = [view.symbols[i] for i in top_n_indices]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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