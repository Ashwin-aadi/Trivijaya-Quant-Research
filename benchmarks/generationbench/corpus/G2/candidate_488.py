from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "High volume on a directional move indicates strong market sentiment. "
        "Stocks with significant volume and a clear price movement are likely to continue trending."
    )

    def __init__(self, window: int = 20, threshold_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._threshold_volume_ratio = threshold_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(pl.col("adj_close").last().alias("close"))
        recent_volumes = history.select(pl.col("volume"))

        breakout_signals = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)

            if symbol_history.height < self._window:
                continue

            latest_close = float(recent_closes[symbol])
            open_price = float(symbol_history.select(pl.col("open").first()).item())
            volume = float(recent_volumes[symbol][-1])

            price_move_ratio = (latest_close - open_price) / abs(open_price)

            if price_move_ratio > 0:
                positive_move = True
            else:
                positive_move = False

            high_volume = volume >= symbol_history.select(pl.col("volume").max()).item() * self._threshold_volume_ratio

            if positive_move and high_volume:
                breakout_signals.append(symbol)

        if not breakout_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest