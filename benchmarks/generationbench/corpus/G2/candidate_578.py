from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility periods are often followed by mean-reverting markets. By entering "
        "positions based on the recent volatility of assets, we can capture this effect and "
        "potentially profit from a return to more normal levels of volatility."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_returns = (history[symbol].to_list()[1:] / history[symbol].shift(1).to_list()[:-1]) - 1.0
            volatility_score = pl.Series(daily_returns).std()
            volatility_scores[symbol] = float(volatility_score)

        sorted_symbols = [symbol for symbol, _ in sorted(volatility_scores.items(), key=lambda x: -x[1])]
        if len(sorted_symbols) < self._threshold:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = sorted_symbols[: int(self._threshold)]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest