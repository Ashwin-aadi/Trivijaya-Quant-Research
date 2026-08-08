from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualSectorStrategy(Strategy):
    rationale = (
        "This strategy leverages the composite performance of stocks in renewable energy and agricultural productivity. "
        "By combining two weakly related characteristics, it aims to capture opportunities where both sectors might see "
        "simultaneous growth driven by economic and policy shifts."
    )

    def __init__(self, window_a: int = 50, window_b: int = 30, top_n: int = 20) -> None:
        self._window_a = window_a
        self._window_b = window_b
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_a, self._window_b))
        if history.height < max(self._window_a, self._window_b):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            recent_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
            if len(recent_closes) < max(self._window_a, self._window_b):
                continue

            # Calculate characteristic A score (50-day moving average of returns)
            returns = [(recent_closes[i+1] - recent_closes[i]) / recent_closes[i] for i in range(len(recent_closes)-1)]
            char_a_score = sum(returns[-self._window_a:]) / self._window_a

            # Calculate characteristic B score (30-day volume-weighted average price)
            volumes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["volume"].drop_nulls().to_list()]
            char_b_score = sum(recent_closes[i] * volumes[i] for i in range(len(recent_closes))) / sum(volumes[-self._window_b:])

            scores[symbol] = 0.6 * char_a_score + 0.4 * char_b_score

        sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        picks = list(sorted_scores.keys())[:self._top_n]

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