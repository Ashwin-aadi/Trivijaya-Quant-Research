from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines momentum with volatility to identify stocks that are both "
        "strongly trending and experiencing high price fluctuations. Strong momentum suggests "
        "positive market sentiment, while high volatility can indicate profit-taking opportunities."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._momentum_window, self._volatility_window))
        if closes.height < max(self._momentum_window, self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_series) < max(self._momentum_window, self._volatility_window):
                continue

            momentum_score = (
                (close_series[-1] - close_series[0]) / sum(abs(x - y) for x, y in zip(close_series[:-1], close_series[1:]))
            )
            volatility_score = pl.col(symbol).std().item()

            momentum_scores[symbol] = momentum_score
            volatility_scores[symbol] = volatility_score

        combined_scores = {s: 0.5 * momentum_scores[s] + 0.5 * volatility_scores[s] for s in momentum_scores}

        top_symbols = sorted(combined_scores, key=combined_scores.get, reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest