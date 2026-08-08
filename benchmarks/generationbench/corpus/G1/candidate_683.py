from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining two weakly related characteristics can sometimes provide a more robust "
        "signal than relying on either one alone. This strategy looks for stocks with both "
        "relatively high momentum and low volatility over the past 20 days."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = _calculate_momentum_score(history)
        volatility_scores = _calculate_volatility_score(history)

        combined_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in momentum_scores or symbol not in volatility_scores:
                continue
            combined_scores[symbol] = (momentum_scores[symbol] + volatility_scores[symbol]) / 2

        sorted_symbols = [s[0] for s in sorted(combined_scores.items(), key=lambda x: -x[1])]
        top_n = sorted_symbols[:5]

        if not top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_momentum_score(history: pl.DataFrame) -> dict[str, float]:
    momentum_scores: dict[str, float] = {}
    for symbol in history.columns[2:]:
        returns = (history[symbol].to_list()[1:] / history[symbol].shift(1).to_list()[:-1]) - 1
        momentum_scores[symbol] = max(returns)
    return momentum_scores


def _calculate_volatility_score(history: pl.DataFrame) -> dict[str, float]:
    volatility_scores: dict[str, float] = {}
    for symbol in history.columns[2:]:
        returns = (history[symbol].to_list()[1:] / history[symbol].shift(1).to_list()[:-1]) - 1
        std_dev = pl.Series(returns).std()
        volatility_scores[symbol] = 1.0 / (std_dev + 1e-6)
    return volatility_scores