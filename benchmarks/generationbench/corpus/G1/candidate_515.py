from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 20-day momentum "
        "and the 50-day moving average. The idea is that stocks with strong short-term "
        "momentum and a positive long-term trend may offer better opportunities."
    )

    def __init__(self, window_20: int = 20, window_50: int = 50) -> None:
        self._window_20 = window_20
        self._window_50 = window_50

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_50 + 1)

        if history.height < self._window_50 + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        moving_average_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(closes) < self._window_50 + 1:
                continue

            close_20 = float(history[symbol][-self._window_20])
            close_50 = float(history[symbol][-self._window_50])

            momentum_score = (close_20 / close_50 - 1.0) * 100
            moving_average_score = 1 if close_20 > close_50 else 0

            momentum_scores[symbol] = momentum_score
            moving_average_scores[symbol] = moving_average_score

        final_scores = {
            symbol: (momentum_scores[symbol] + moving_average_scores[symbol]) / 2.0
            for symbol in momentum_scores.keys()
            if symbol in moving_average_scores and momentum_scores[symbol] > 0
        }

        sorted_symbols = [
            s for s, _ in sorted(final_scores.items(), key=lambda item: -item[1])
        ][:5]

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest