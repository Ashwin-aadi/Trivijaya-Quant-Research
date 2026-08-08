from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsMomentumStrategy(Strategy):
    rationale = (
        "This strategy exploits the composite characteristic of earnings quality and momentum. "
        "It identifies stocks with positive earnings revisions and recent strong price trends, "
        "hoping to capture value from both fundamental and technical indicators."
    )

    def __init__(self, earnings_window: int = 90, momentum_window: int = 183) -> None:
        self._earnings_window = earnings_window
        self._momentum_window = momentum_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window)
        if history.is_empty() or history.height < self._momentum_window:
            return Signal(information_available_at=stamp, weights={})

        # Placeholder for earnings revisions data (could be exogenously provided)
        earnings_revisions = {
            "Symbol1": [0.2, 0.1, -0.3, 0.5],
            "Symbol2": [-0.1, 0.4, 0.6, -0.2],
            # Add more symbols and data as needed
        }

        # Calculate momentum score for each symbol
        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in earnings_revisions or history.height < self._momentum_window:
                continue

            close_prices = [float(v) for v in history[symbol].to_list()[-self._momentum_window:]]
            momentum_score = (close_prices[-1] - close_prices[0]) / close_prices[0]
            momentum_scores[symbol] = momentum_score

        # Rank symbols by earnings revisions and momentum scores
        ranked_symbols = sorted(
            earnings_revisions.keys(),
            key=lambda s: (
                max(earnings_revisions.get(s, [0])) if s in earnings_revisions else -1,
                momentum_scores.get(s, -1)
            ),
            reverse=True,
        )[:20]

        weight = 1.0 / len(ranked_symbols) if ranked_symbols else 0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in ranked_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest