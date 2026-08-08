from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "We seek stocks that have both strong relative strength and high volume momentum. "
        "Strong relative strength suggests a stock is performing well compared to the market, "
        "while high volume momentum indicates recent buying pressure. Together, these characteristics "
        "can indicate a stock poised for continued upward movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_momentum_scores = []
        for symbol in view.symbols:
            hist = view.history(symbol=symbol, lookback=self._window)
            adj_closes = hist["adj_close"].to_list()
            volumes = hist["volume"].to_list()

            if len(adj_closes) < self._window or len(volumes) < self._window:
                continue

            relative_strength = (
                pl.Series(adj_closes[-1]) / pl.Series(adj_closes[0]) - 1
            ).mean()
            volume_momentum_score = sum(
                v for _, v in zip(adj_closes, volumes) if v > 1.5 * max(volumes)
            ) / len(adj_closes)

            volume_momentum_scores.append(volume_momentum_score)

        top_symbols = [
            symbol
            for symbol, score in sorted(
                zip(view.symbols, volume_momentum_scores), key=lambda x: x[1], reverse=True
            )
            if relative_strength > 0.2 and score > 0.5
        ][:5]

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