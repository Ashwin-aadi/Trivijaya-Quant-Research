from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks with the highest relative performance "
        "over a short period are likely to continue outperforming in the near future. By "
        "identifying and investing in these top performers, we can potentially capture positive "
        "returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        returns: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            latest_price = float(closes[stamp, symbol])
            avg_price = sum(prices[-20:]) / len(prices[-20:])
            return_ratio = latest_price / avg_price - 1.0
            returns.append(return_ratio)

        top_indices = sorted(range(len(returns)), key=lambda i: returns[i], reverse=True)[: self._top_n]
        picks = [view.symbols[i] for i in top_indices]

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