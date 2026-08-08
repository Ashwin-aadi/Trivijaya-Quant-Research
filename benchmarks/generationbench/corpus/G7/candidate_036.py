from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks with strong recent price trends and "
        "allocates capital to those names, capturing the collective market sentiment towards "
        "positive price movements."
    )

    def __init__(self, window: int = 60, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].to_list()]
            recent_closes = close_prices[-self._window:]
            daily_returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1] if i > 0 else 0.0 for i in range(len(recent_closes))]
            momentum_score = sum(daily_returns) / self._window
            momentum_scores[symbol] = momentum_score

        top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest