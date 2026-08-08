from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment and can indicate "
        "the beginning of a trend. By identifying symbols with significant volume on a day when "
        "prices move in a specific direction, we can generate profitable trades."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        if latest_closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            session_dates = [str(d) for d in history["session_date"].to_list()]
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volumes = [int(v) for v in history[f"{symbol}_volume"].drop_nulls().to_list()]

            if len(adj_closes) < self._window + 1:
                continue

            # Calculate the price change and volume on the last session
            last_close = adj_closes[-1]
            last_volume = volumes[-1]

            # Check for a significant move and confirm with high volume
            if (
                (last_close - adj_closes[-2] > 0 and last_volume > sum(volumes[:-1]))
                or (last_close - adj_closes[-2] < 0 and last_volume < sum(volumes[:-1]))
            ):
                picks.append(symbol)

        picks = picks[:5]
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