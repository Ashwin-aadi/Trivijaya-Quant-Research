from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to follow trends by identifying stocks with recent high volatility "
        "and adjusting weights accordingly. High volatility indicates a strong trend that can be leveraged."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        volatility_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high = float(history[symbol]["high"].max())
            low = float(history[symbol]["low"].min())
            close = float(history[symbol]["close"][-1])
            volatility_score = abs((high - low) / close)
            volatility_scores[symbol] = volatility_score

        top_symbols = sorted(volatility_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest