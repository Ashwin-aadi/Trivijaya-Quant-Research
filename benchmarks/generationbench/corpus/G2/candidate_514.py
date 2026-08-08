from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Seasonal effects in the Indian market can arise due to specific economic activities "
        "and weather patterns. For instance, certain sectors like agriculture may have higher "
        "activity during monsoon seasons, leading to predictable price movements."
    )

    def __init__(self, season: str = "monsoon", window: int = 30) -> None:
        self._season = season
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_symbols = {
            "agriculture": ["NIFTY AGRI", "NIFTY AGRIBANK"],
            "retail": ["NIFTY RETAIL", "NIFTY FMCG"],
        }

        if self._season not in seasonal_symbols:
            return Signal(information_available_at=stamp, weights={})

        symbols = seasonal_symbols[self._season]
        symbol_data = {symbol: closes[symbol] for symbol in symbols}

        seasonal_effect_scores = {
            symbol: float(closes[-1][symbol]) / pl.col(symbol).mean().over("session_date")
            - 1.0
            for symbol in symbols
        }

        sorted_symbols = [
            symbol for _, symbol in sorted(seasonal_effect_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        top_symbol = sorted_symbols[0]
        weight = 1.0

        return Signal(
            information_available_at=stamp, weights={top_symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest