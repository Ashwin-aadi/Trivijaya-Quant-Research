from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion identifies stocks that have moved far from their mean price and "
        "predicts they will move back. Short-horizon mean reversion looks at recent prices "
        "to identify extreme deviations over a 5-day window."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            rolling_mean = sum(values[-self._window:]) / self._window
            current_price = values[-1]
            score = (current_price - rolling_mean) / rolling_mean

            if abs(score) > 0.2:  # Consider stocks with a score greater than ±20% from mean as candidates
                mean_reversion_scores[symbol] = score

        picks: list[str] = [s for s, _ in sorted(mean_reversion_scores.items(), key=lambda item: abs(item[1]), reverse=True)]
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