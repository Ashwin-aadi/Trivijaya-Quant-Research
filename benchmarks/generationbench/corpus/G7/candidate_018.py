from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength60d(Strategy):
    rationale = (
        "Relative strength over a 60-day period measures how well a stock is performing "
        "compared to its peers. A higher relative strength indicates that the stock has outperformed "
        "the broader market and may be undervalued, making it a potential buy."
    )

    def __init__(self, window: int = 60, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._window:
                continue

            avg_close = sum(recent_closes) / len(recent_closes)
            relative_strength = recent_closes[-1] / avg_close - 1
            relative_strengths.append((symbol, relative_strength))

        sorted_strengths = sorted(relative_strengths, key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_strengths[: self._top_n]]
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