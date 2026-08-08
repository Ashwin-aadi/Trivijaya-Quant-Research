from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SentimentGDPComposite(Strategy):
    rationale = (
        "This strategy leverages a composite score combining daily sentiment analysis "
        "of social media trends and macroeconomic GDP growth rates to identify trading signals."
    )

    def __init__(self, window: int = 20, threshold_buy: float = 0.5, threshold_sell: float = 0.3) -> None:
        self._window = window
        self._threshold_buy = threshold_buy
        self._threshold_sell = threshold_sell

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sentiment_scores = [float(s) for s in _sentiment_scores(history)]
        gdp_growth_rates = [float(g) for g in _gdp_growth_rates(history)]

        if len(sentiment_scores) < self._window or len(gdp_growth_rates) < self._window:
            return Signal(information_available_at=stamp, weights={})

        composite_scores = [(s + g) / 2.0 for s, g in zip(sentiment_scores, gdp_growth_rates)]

        ranked_symbols = sorted(view.symbols, key=lambda x: composite_scores[view.closes(lookback=self._window).column(x).to_list()[-1]], reverse=True)
        
        if composite_scores[-1] > self._threshold_buy:
            weights = {s: 0.05 for s in ranked_symbols[:20]}
            return Signal(information_available_at=stamp, weights=weights)

        elif composite_scores[-1] < self._threshold_sell:
            weights = {s: -0.05 for s in ranked_symbols[:20]}
            return Signal(information_available_at=stamp, weights=weights)

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _sentiment_scores(history: pl.DataFrame) -> list[float]:
    # Placeholder function for sentiment score calculation
    return [1.0] * history.height


def _gdp_growth_rates(history: pl.DataFrame) -> list[float]:
    # Placeholder function for GDP growth rate data retrieval
    return [0.5] * history.height