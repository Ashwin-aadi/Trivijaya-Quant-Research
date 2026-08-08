from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy identifies and invests in stocks with the strongest relative momentum "
        "over a specified period. The idea is to capture the excess returns associated with "
        "stocks that have performed well recently."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._window:
                continue

            # Calculate daily returns
            returns = [(recent_closes[i] / recent_closes[i - 1] - 1.0) for i in range(1, self._window)]
            avg_return = sum(returns) / self._window

            momentum_scores[symbol] = avg_return

        # Sort by momentum score and pick top N
        sorted_symbols = [s for s, v in sorted(momentum_scores.items(), key=lambda item: -item[1])]
        picks = sorted_symbols[: self._lookback]

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