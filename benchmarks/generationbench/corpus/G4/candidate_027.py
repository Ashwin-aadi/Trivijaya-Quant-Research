from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the liquidity anomaly in Indian equity markets by "
        "screening for stocks based on their trading activity and equally weighting "
        "the selected assets. The economic reasoning behind this is that less liquid "
        "stocks often misprice, offering higher returns due to lower demand from "
        "trading constraints."
    )

    def __init__(self, window: int = 30, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        turnover_ratio = (history["volume"] * 2) / (
            (history["high"] - history["low"]) * view.symbols.count()
        )
        liquidity_scores = turnover_ratio.to_list()

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_symbols = [
            symbol
            for symbol, score in zip(symbols, liquidity_scores)
            if score <= self._threshold
        ]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest