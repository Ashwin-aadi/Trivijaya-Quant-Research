from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relatively to their peers in recent periods to continue outperforming over a longer "
        "term. By identifying and investing in such stocks, we can capitalize on this effect."
    )

    def __init__(self, lookback_window: int = 30, top_n: int = 5) -> None:
        self._lookback_window = lookback_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window:
            return Signal(information_available_at=stamp, weights={})

        # Compute the relative performance of each stock
        relative_performances = []
        for symbol in view.symbols:
            if symbol not in closes.columns or len(closes[symbol].to_list()) < self._lookback_window:
                continue

            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            recent_close = close_values[-1]
            min_price = min(close_values)
            max_price = max(close_values)

            # Calculate the relative performance as the percentage change from the lowest to highest price
            if max_price == min_price:
                continue

            rel_perf = (recent_close - min_price) / (max_price - min_price)
            relative_performances.append((symbol, rel_perf))

        # Sort by relative performance in descending order and pick top N performers
        sorted_performances = sorted(relative_performances, key=lambda x: x[1], reverse=True)[: self._top_n]
        picks = [s[0] for s in sorted_performances]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={p: weight for p in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest