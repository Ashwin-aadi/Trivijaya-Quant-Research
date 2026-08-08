from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency for asset prices to revert "
        "to previous price levels after extreme movements. By identifying symbols that have"
        " moved significantly from their recent average price, we can generate signals for "
        "potential reversals."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = history["adj_close"].mean().item()
        symbol_reversions: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            recent_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(recent_closes) < self._window:
                continue
            latest_close = float(history.filter(pl.col("symbol") == symbol).select("adj_close").tail(1).item())
            price_deviation = (latest_close - avg_close) / avg_close
            if abs(price_deviation) >= self._threshold:
                symbol_reversions[symbol] = price_deviation

        if not symbol_reversions:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbol_reversions)
        return Signal(
            information_available_at=stamp,
            weights={s: abs(weight) for s in symbol_reversions.keys()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest