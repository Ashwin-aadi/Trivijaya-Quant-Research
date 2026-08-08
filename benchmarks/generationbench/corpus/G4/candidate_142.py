from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Exploiting seasonality in the Indian equity market by identifying specific months or "
        "quarters where historical data shows higher returns. This strategy focuses on sectors "
        "known to outperform during certain periods, such as real estate before monsoons."
    )

    def __init__(self, window: int = 10, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]

        seasonal_scores = {}
        for symbol in symbols:
            values = [float(v) for v in history.filter(pl.col("session_date").dt.month() == 7)[
                f"{symbol}"
            ].to_list()]
            avg_return = sum(values) / len(values)
            recent_return = (closes[symbol].max().item() - closes[symbol].min().item()) / (
                closes[symbol].min().item()
            )
            seasonal_scores[symbol] = recent_return - avg_return

        sorted_scores = sorted(seasonal_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_scores[: self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest