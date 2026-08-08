from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy aims to exploit a combination of technical and fundamental signals. "
        "Specifically, it targets stocks with strong recent price momentum and positive earnings "
        "surprises, which often indicate strong underlying fundamentals."
    )

    def __init__(self, window: int = 10, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        mean_close = closes.mean()
        momentum = (closes[-1] / mean_close - 1.0).to_list()[0]

        earnings_surprises = _calculate_earnings_surprise(view)
        if earnings_surprises.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [symbol for symbol in view.symbols if symbol in earnings_surprises]
        top_symbols.extend([symbol for symbol in history["symbol"] if symbol not in top_symbols and
                            momentum > 0.1])
        picks = top_symbols[: self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_earnings_surprise(view: MarketView) -> pl.DataFrame:
    history = view.history(lookback=365)
    if history.is_empty():
        return pl.DataFrame()

    earnings_announcements = {symbol: float(value) for symbol, value in
                              zip(history["symbol"], history["adj_close"].to_list())}

    actual_earnings = {symbol: latest_close() for symbol in view.symbols}
    surprises = [(symbol, actual - expected)
                 for symbol, expected in earnings_announcements.items()
                 if symbol in actual_earnings and
                 abs(actual_earnings[symbol] - expected) > 0.1]

    return pl.DataFrame(surprises, schema=["symbol", "surprise"])