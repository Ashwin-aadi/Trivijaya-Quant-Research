from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the strength of a stock's "
        "recent momentum and its valuation relative to historical levels. A strong momentum and "
        "a low valuation are indicators that the stock may be undervalued but showing signs of "
        "strength, potentially leading to future gains."
    )

    def __init__(self, momentum_window: int = 20, valuation_window: int = 50) -> None:
        self._momentum_window = momentum_window
        self._valuation_window = valuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._valuation_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        valuation_scores: dict[str, float] = {}

        for symbol in view.symbols:
            recent_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._momentum_window:
                continue
            momentum_score = (recent_closes[-1] - recent_closes[0]) / sum(
                abs(r - recent_closes[0]) for r in recent_closes
            )
            momentum_scores[symbol] = momentum_score

        valuation_history = view.closes(lookback=self._valuation_window)
        for symbol in view.symbols:
            if symbol not in valuation_history.columns:
                continue
            values = [float(v) for v in valuation_history[symbol].drop_nulls().to_list()]
            if len(values) < self._valuation_window:
                continue
            min_value, max_value = min(values), max(values)
            recent_close = float(view.latest_close()[symbol])
            if recent_close >= 0.9 * min_value and recent_close <= 1.1 * max_value:
                valuation_score = (recent_close - min_value) / (max_value - min_value)
            else:
                valuation_score = 0.0
            valuation_scores[symbol] = valuation_score

        combined_scores = {
            symbol: momentum_scores.get(symbol, 0.0) + valuation_scores.get(symbol, 0.0)
            for symbol in view.symbols
        }

        sorted_symbols = [
            s for _, s in sorted(combined_scores.items(), key=lambda item: -item[1])
        ][:5]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest