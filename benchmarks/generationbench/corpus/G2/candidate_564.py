from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for assets that have outperformed "
        "in recent periods to continue to outperform. This strategy buys top performers and "
        "sells underperformers based on their relative performance over a lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        rank_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            returns = [(values[i + 1] - values[i]) / values[i] for i in range(len(values[:-1]))]
            rank_scores[symbol] = sum(returns) / (len(returns))

        top_symbols = sorted(rank_scores, key=rank_scores.get, reverse=True)[:5]
        bottom_symbols = sorted(rank_scores, key=rank_scores.get)[:5]

        weight_top = 0.2
        weight_bottom = -0.2
        weights = {s: weight_top for s in top_symbols}
        for b in bottom_symbols:
            if b not in weights:
                weights[b] = weight_bottom

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w != 0.0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest