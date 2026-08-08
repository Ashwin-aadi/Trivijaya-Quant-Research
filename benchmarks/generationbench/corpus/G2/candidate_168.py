from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the recent past to continue outperforming over a short horizon. This is based on the "
        "idea that market sentiment and investor behavior tend to persist."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_prices) < self._window:
                continue

            # Compute the momentum score as the percentage change from the earliest to the latest close.
            momentum_score = (close_prices[-1] - close_prices[0]) / close_prices[0]
            momentum_scores[symbol] = momentum_score

        sorted_symbols = [
            symbol for _, symbol in sorted(momentum_scores.items(), key=lambda item: -item[1])
        ][:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest