from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy leverages a composite of two signals: one based on high momentum "
        "and another based on value. The idea is that combining these can provide a more "
        "robust signal for entry."
    )

    def __init__(self, momentum_window: int = 10, value_threshold: float = 0.5) -> None:
        self._momentum_window = momentum_window
        self._value_threshold = value_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + 1)

        if history.height < self._momentum_window + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "adj_close" not in history[symbol]:
                continue

            adj_closes = [float(v) for v in history[symbol]["adj_close"].drop_nulls().to_list()]
            if len(adj_closes) < self._momentum_window + 1:
                continue
            momentum_score = (adj_closes[-1] - adj_closes[0]) / sum(
                adj_closes[i] > adj_closes[i + 1] for i in range(len(adj_closes) - 1)
            )
            if momentum_score >= self._value_threshold:
                momentum_scores[symbol] = momentum_score

        value_scores = {}
        closes = view.closes(lookback=self._momentum_window)
        for symbol in view.symbols:
            if symbol not in closes.columns or "adj_close" not in closes[symbol]:
                continue
            adj_closes = [float(v) for v in closes[symbol].to_list()]
            value_score = min(adj_closes) / sum(
                adj_closes[i] < adj_closes[i + 1] for i in range(len(adj_closes) - 1)
            )
            if value_score >= self._value_threshold:
                value_scores[symbol] = value_score

        final_selections = []
        combined_scores = {}
        for symbol in momentum_scores.keys() & value_scores.keys():
            combined_scores[symbol] = (momentum_scores[symbol] + value_scores[symbol]) / 2
            if combined_scores[symbol] > self._value_threshold:
                final_selections.append(symbol)

        weights = {symbol: 1.0 / len(final_selections) for symbol in final_selections}
        return Signal(
            information_available_at=stamp, weights={s: weights[s] for s in final_selections if s in weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest