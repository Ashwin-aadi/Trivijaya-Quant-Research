from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the market index "
        "can lead to outperformance. Stocks that consistently outperform the benchmark "
        "are likely to continue this trend in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        market_index = "NIFTY100"  # Assuming NIFTY100 as the benchmark
        closes = view.closes(lookback=self._window)

        relative_strength_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol == market_index or symbol not in history.columns:
                continue

            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window:
                continue
            market_close_values = [
                float(v) for v in history[market_index].drop_nulls().to_list()
            ]
            if len(market_close_values) < self._window:
                continue

            daily_returns_symbol = [(close / prev_close - 1.0) for close, prev_close in zip(close_values[1:], close_values[:-1])]
            daily_returns_market = [(close / prev_close - 1.0) for close, prev_close in zip(market_close_values[1:], market_close_values[:-1])]

            avg_return_symbol = sum(daily_returns_symbol) / len(daily_returns_symbol)
            avg_return_market = sum(daily_returns_market) / len(daily_returns_market)

            relative_strength_score = (avg_return_symbol - avg_return_market) / abs(avg_return_market)
            relative_strength_scores[symbol] = relative_strength_score

        if not relative_strength_scores:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = sorted(relative_strength_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest