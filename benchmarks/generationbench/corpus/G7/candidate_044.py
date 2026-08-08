from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment and "
        "can lead to sustainable price movements. By focusing on the last 3 hours of trading, "
        "we can identify such moves with confidence."
    )

    def __init__(self, window: int = 3) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volume_changes: dict[str, float] = {}
        for symbol in symbols:
            adj_close_series = history[symbol].select("adj_close").to_numpy().flatten()
            volume_series = history[symbol].select("volume").to_numpy().flatten()

            if len(adj_close_series) < self._window * 2 or len(volume_series) < self._window * 2:
                continue

            # Calculate VWAP for the last window periods
            vwap_changes = []
            for i in range(self._window, len(adj_close_series), 1):
                volume_sum = sum(volume_series[i - self._window : i])
                if volume_sum == 0:
                    continue
                vwap_change = (adj_close_series[i] - adj_close_series[i - self._window]) / adj_close_series[i - self.window]
                vwap_changes.append(vwap_change * volume_series[i])

            if not vwap_changes:
                continue

            avg_vwap_change = sum(vwap_changes) / sum(volume_series[-self._window:])
            volume_changes[symbol] = avg_vwap_change

        top_symbols = sorted(volume_changes, key=volume_changes.get, reverse=True)[:3]
        weights = {s: 1.0 / 3 for s in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] for s in view.symbols if s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest