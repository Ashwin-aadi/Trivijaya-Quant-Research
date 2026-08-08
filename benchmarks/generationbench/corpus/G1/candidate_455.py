from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeFeatureStrategy(Strategy):
    rationale = (
        "Combining the 20-day volatility with the recent price trend can provide a more nuanced view of stock "
        "behavior. High volatility coupled with an upward trend suggests increased buying interest."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()
        volatility_scores = {}
        for symbol in view.symbols:
            if symbol not in latest_close.keys():
                continue

            session_dates = history["session_date"].to_list()[1:]
            closes = [latest_close[symbol]] + [float(c) for c in history[symbol].drop_nulls().to_list()]
            r = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            volatility_score = max(abs(v) for v in r)
            volatility_scores[symbol] = volatility_score

        top_symbols = sorted(volatility_scores.items(), key=lambda x: (x[1], -latest_close[x[0]]), reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest