from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion is a phenomenon where stock prices that have deviated significantly from "
        "their historical mean revert back towards the average. In short-horizon trading, this can "
        "be used to identify stocks that are overbought or oversold and make profitable trades by "
        "buying on dips or selling on rallies."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "adj_close" not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].to_list()]
            if len(adj_closes) < self._window:
                continue

            mean_adj_close = sum(adj_closes[-self._window:]) / self._window
            last_adj_close = adj_closes[-1]
            score = (last_adj_close - mean_adj_close) / mean_adj_close
            mean_reversion_scores[symbol] = score

        top_symbols = sorted(mean_reversion_scores, key=mean_reversion_scores.get, reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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