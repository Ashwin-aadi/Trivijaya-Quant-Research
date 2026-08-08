from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit strong seasonality, with prices "
        "tending to rise or fall around specific times of the year. By identifying these "
        "patterns, we can generate profitable trades based on historical price movements."
    )

    def __init__(self, window: int = 365, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            symbol_data = history.select(pl.col("symbol").eq(symbol)).to_dict(False)
            closes = [float(v) for v in symbol_data[0]["adj_close"]]
            max_close = max(closes)
            min_close = min(closes)
            seasonality[symbol] = (max_close - min_close) / self._window

        sorted_symbols = sorted(seasonality.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_symbols[: self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest