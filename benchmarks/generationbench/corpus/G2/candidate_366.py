from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Companies that consistently outperform their peers in terms of return may continue "
        "to do so due to superior management, better business models, or market positioning. "
        "Identifying such companies early can lead to higher returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = {s: float(v) for s, v in zip(view.symbols, closes["session_date", 0].to_list())}
        rank_scores = []
        for symbol in symbols:
            returns = [float(v) for v in (closes[symbol] / closes[symbol].shift(1) - 1.0).drop_nulls().to_list()]
            if len(returns) < self._window // 2:
                continue
            rank_score = sum(returns[-self._window // 2:]) / (self._window // 2)
            rank_scores.append((symbol, rank_score))

        ranked_symbols = sorted(rank_scores, key=lambda x: x[1], reverse=True)
        top_5_symbols = [s for s, _ in ranked_symbols[:5]]
        weight = 1.0 / len(top_5_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_5_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest