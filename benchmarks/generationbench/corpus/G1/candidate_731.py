from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy leverages a combination of short-term momentum and value "
        "signals to identify undervalued stocks with strong recent performance. "
        "By combining these signals, the strategy aims to capture both overbought and "
        "oversold conditions in the market."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        value_scores: dict[str, float] = {}

        for symbol in view.symbols:
            closes = [float(v) for v in view.closes(lookback=self._window)[symbol].to_list()]
            if len(closes) < self._window:
                continue

            # Calculate momentum score
            close_ratio = (closes[-1] - closes[0]) / abs(closes[0])
            momentum_scores[symbol] = close_ratio

            # Calculate value score
            low_high_diff = max(closes) - min(closes)
            value_scores[symbol] = 1.0 - ((closes[-1] - min(closes)) / low_high_diff)

        final_scores: dict[str, float] = {
            symbol: momentum_scores[symbol] * value_scores[symbol]
            for symbol in view.symbols
        }

        picks: list[str] = sorted(final_scores, key=final_scores.get, reverse=True)[: self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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