from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Capitalizes on cross-sectional momentum to identify stocks with recent outperformance. "
        "The strategy selects top performers for long-term exposure and exits based on absolute or relative loss."
    )

    def __init__(self, window: int = 60, threshold: float = 0.25, max_positions: int = 30) -> None:
        self._window = window
        self._threshold = threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = [
            (
                (float(closes[symbol].max().item()) / float(history.select(pl.col("close").first()).to_series()[0]) - 1)
            )
            for symbol in symbols
        ]

        threshold_value = sorted(momentum_scores)[-int(len(symbols) * self._threshold)]
        top_symbols = [symbol for i, symbol in enumerate(symbols) if momentum_scores[i] >= threshold_value]

        if len(top_symbols) > self._max_positions:
            top_symbols = top_symbols[: self._max_positions]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, pl.Date)
    return newest