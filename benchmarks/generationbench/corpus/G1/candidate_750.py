from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength outperform the broader market over time. "
        "This strategy aims to identify and allocate capital to such stocks."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        avg_close = closes.mean().drop_nulls()
        strengths: list[tuple[str, float]] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            strength_ratio = latest_close / avg_close[symbol]
            strengths.append((symbol, strength_ratio))

        strengths.sort(key=lambda x: x[1], reverse=True)
        top_n_strengths = strengths[:5]

        if not top_n_strengths:
            return Signal(information_available_at=stamp, weights={})

        total_weight = 1.0
        weight_per_symbol = total_weight / len(top_n_strengths)
        weights = {symbol: weight_per_symbol for symbol, _ in top_n_strengths}

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest