from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "Combining simple moving average crossovers and relative strength provides a more "
        "robust entry signal by leveraging both trend and momentum indicators."
    )

    def __init__(self, short_window: int = 50, long_window: int = 200) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._short_window, self._long_window))
        if closes.height < max(self._short_window, self._long_window):
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            short_sma = closes[symbol].mean().item()
            long_sma = (
                closes[symbol]
                .sort("session_date")
                .tail(self._long_window)
                .mean()
                .item()
            )
            rsi = _calculate_rsi(closes=symbol, lookback=self._short_window)

            if short_sma > long_sma and rsi < 30:
                signals[symbol] = 1.0

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in signals.items() if w},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_rsi(closes: str, lookback: int) -> float:
    delta = view.closes().select(
        pl.col(closes).shift(-1) - pl.col(closes).shift(0)
    ).drop_nulls()
    up = (delta[delta.ge(0)].mean() * 100).item()
    down = (-delta[delta.le(0)].mean() * 100).item()
    return float(up / (up + down) if up + down > 0 else 0)