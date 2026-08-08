from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks with strong relative "
        "performance over a recent period to continue outperforming in the near future. "
        "This phenomenon is supported by academic research suggesting that market "
        "efficiency is not perfect, and some stocks can exhibit persistent performance "
        "gradients."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = (history["close"] / history["adj_close"].shift(self._window) - 1).alias("momentum")
        history = history.with_columns(momentum_scores)
        history = history.sort(by="momentum", descending=True)

        top_symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        top_symbols = top_symbols[: self._top_n]

        weights = {s: 1.0 / len(top_symbols) for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest