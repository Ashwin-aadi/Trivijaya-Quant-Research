from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies the strongest performers in the market over a "
        "given lookback period and allocates capital to those stocks. This strategy leverages "
        "the principle that past winners tend to continue outperforming."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        rank_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].to_list()]
            if len(close_values) < self._window:
                continue
            # Calculate daily returns and rank them
            returns = [(close_values[i] - close_values[i-1]) / close_values[i-1] for i in range(1, self._window)]
            rank_scores[symbol] = sum(returns)

        top_symbols = sorted(rank_scores.keys(), key=lambda s: rank_scores[s], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest