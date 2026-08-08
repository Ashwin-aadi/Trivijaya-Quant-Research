from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion assumes that asset prices and historical returns will eventually "
        "move back towards their long-term mean. By identifying stocks that have deviated "
        "significantly from this mean in the short term, we can exploit this tendency for "
        "profit."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            if len(prices) < self._window:
                continue

            mean = sum(prices) / len(prices)
            deviation = abs((prices[-1] - mean) / mean)

            if deviation >= self._threshold:
                mean_reversion_scores[symbol] = deviation

        picks: list[str] = sorted(mean_reversion_scores.keys(), key=lambda s: mean_reversion_scores[s])[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest