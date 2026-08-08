from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "We hypothesize that stocks with high short-term momentum and low volatility are "
        "more likely to continue their recent trends. High momentum indicates strong buying "
        "pressure, while low volatility suggests the stock price is stable and less prone to "
        "sharp declines."
    )

    def __init__(self, window_momentum: int = 10, window_volatility: int = 20) -> None:
        self._window_momentum = window_momentum
        self._window_volatility = window_volatility

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_volatility + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].to_list()]
            momentum_score = (closes[-1] - closes[0]) / self._window_momentum
            volatility_score = pl.col(symbol).std().item()
            momentum_scores[symbol] = momentum_score
            volatility_scores[symbol] = volatility_score

        # Select top 5 based on combined score
        combined_scores = {
            s: (momentum_scores.get(s, 0) + 1.0 / (volatility_scores.get(s, 1))) for s in view.symbols
        }
        sorted_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        picks = [symb for symb, _ in sorted_symbols[:5]]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest