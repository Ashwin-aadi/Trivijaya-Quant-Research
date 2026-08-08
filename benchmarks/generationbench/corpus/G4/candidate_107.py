from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsMomentum(Strategy):
    rationale = (
        "This strategy identifies stocks with recent positive earnings surprises and "
        "historical strong momentum. By combining these two weakly related characteristics, "
        "it aims to capture both reactive and persistent stock behaviors."
    )

    def __init__(self, es_window: int = 3, mom_window: int = 12) -> None:
        self._es_window = es_window
        self._mom_window = mom_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._es_window, self._mom_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        es_scores = self._compute_earnings_surprise(history)
        mom_scores = self._compute_momentum(history)

        ranked_symbols = sorted(
            es_scores.keys(),
            key=lambda s: (es_scores[s], -mom_scores[s]),
            reverse=True,
        )[:20]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in ranked_symbols}
        )

    def _compute_earnings_surprise(self, history: pl.DataFrame) -> dict[str, float]:
        es_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            actuals = [float(v) for v in history[f"{symbol}_actual"].to_list()]
            estimates = [float(v) for v in history[f"{symbol}_estimate"].to_list()]

            es_score = sum(a - e for a, e in zip(actuals[-self._es_window:], estimates[-self._es_window:])) / self._es_window
            if es_score > 0:
                es_scores[symbol] = es_score

        return es_scores

    def _compute_momentum(self, history: pl.DataFrame) -> dict[str, float]:
        mom_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[f"{symbol}_close"].to_list()]

            mom_score = (closes[-1] - closes[0]) / closes[0]
            if mom_score > 0:
                mom_scores[symbol] = mom_score

        return mom_scores


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest