from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines volume spikes with price momentum to identify potentially "
        "overbought or oversold conditions in the market. Volume spikes can indicate short-term "
        "liquidity imbalances, while strong price momentum suggests continued interest from "
        "traders. By looking for both, we aim to capture opportunities where these signals align."
    )

    def __init__(self, volume_window: int = 20, momentum_window: int = 10) -> None:
        self._volume_window = volume_window
        self._momentum_window = momentum_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._volume_window, self._momentum_window))
        if closes.height < max(self._volume_window, self._momentum_window):
            return Signal(information_available_at=stamp, weights={})

        volume_history = view.history(lookback=self._volume_window)
        momentum_history = view.history(lookback=self._momentum_window)

        high_volume_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in volume_history.symbol.to_list():
                continue
            volume_values = [float(v) for v in volume_history.filter(pl.col("symbol") == symbol)[
                "volume"].to_list()]
            if len(volume_values) < self._volume_window:
                continue
            if max(volume_values) >= 1.5 * pl.col("volume").mean().over("session_date"):
                high_volume_symbols.append(symbol)

        strong_momentum_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in momentum_history.symbol.to_list():
                continue
            price_changes = [float(c - o) / o for o, c in zip(
                momentum_history.filter(pl.col("symbol") == symbol)["open"].to_list(),
                momentum_history.filter(pl.col("symbol") == symbol)["close"].to_list()
            )]
            if sum(price_changes[-self._momentum_window:]) > 0.1 * len(price_changes):
                strong_momentum_symbols.append(symbol)

        combined_signals = set(high_volume_symbols).intersection(set(strong_momentum_symbols))
        if not combined_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in combined_signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest