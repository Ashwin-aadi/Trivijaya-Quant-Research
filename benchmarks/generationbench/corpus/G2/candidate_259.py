from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when volatility is low and prices move in a narrow range. "
        "During such periods, the probability of an breakout increases, as tight ranges indicate "
        "accumulated buying or selling pressure that may soon be exhausted. Investing in symbols "
        "with high dispersion relative to their recent price ranges can thus provide opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        recent_highs = (
            history.select("high")
            .sort("session_date", descending=True)
            .head(1)
            .to_dict(False)[0]["high"]
        )
        recent_lows = (
            history.select("low")
            .sort("session_date", descending=True)
            .head(1)
            .to_dict(False)[0]["low"]
        )

        recent_ranges = [recent_high - recent_low for recent_high, recent_low in zip(recent_highs, recent_lows)]
        mean_range = sum(recent_ranges) / len(recent_ranges)

        dispersion_scores = [
            float((history[pl.col("symbol") == symbol].select(pl.col("adj_close")).to_list()[0][0] - min(history.select("low")) + max(history.select("high"))) / (2 * mean_range))
            for symbol in symbols
        ]

        top_symbols = sorted(zip(symbols, dispersion_scores), key=lambda x: x[1], reverse=True)[:5]

        weights = {symbol: 1.0 / len(top_symbols) for symbol, _ in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest