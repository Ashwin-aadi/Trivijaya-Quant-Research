from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of stock prices to revert to their "
        "mean level over time. A short-horizon mean reversion strategy can profit from "
        "overreacted price movements."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if any(symbol not in closes.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_close = (history["adj_close"] / self._window).sum().item()
        reversion_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            symbol_closes = [float(v) for v in closes[symbol].to_list()]
            if len(symbol_closes) < self._window:
                continue

            recent_close = symbol_closes[-1]
            reversion_signals[symbol] = (recent_close - mean_close) / recent_close

        sorted_signals = sorted(reversion_signals.items(), key=lambda x: abs(x[1]))
        top_symbols = [symbol for symbol, _ in sorted_signals[:5]]
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