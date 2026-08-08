from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-term price deviations from historical averages are driven by noise trader behavior "
        "and market inefficiencies. Stocks that deviate significantly from their 20-day moving average "
        "are likely to revert to their mean prices in the short term, providing trading opportunities."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = history["symbol"].to_list()
        closes = {symbol: [] for symbol in symbols}
        for symbol in symbols:
            series = (
                history.filter(pl.col("symbol") == symbol)["adj_close"]
                .sort(by="session_date")
                .to_series()
            )
            closes[symbol] = [float(v) for v in series.to_list()]

        sma_values = []
        for symbol, close_values in closes.items():
            if len(close_values) < self._window:
                continue
            sma = sum(close_values[-self._window:]) / self._window
            sma_values.append((symbol, sma))

        sma_values.sort(key=lambda x: (x[1] - float(view.latest_close()[x[0]])), reverse=True)
        top_symbols = [s for s, _ in sma_values[: self._top_n]]

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