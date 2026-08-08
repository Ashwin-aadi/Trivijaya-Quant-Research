from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Directional moves are more likely to be sustained when they are accompanied by "
        "increased volume. This strategy identifies symbols where the recent close is higher "
        "than the previous close and has been consistently increasing in volume over the past 10 sessions."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volumes = [int(v) for v in history["volume"].drop_nulls().to_list()]

            if (
                len(adj_closes) < self._window + 2 or
                adj_closes[-1] <= adj_closes[-2] or
                sum(volumes[i] > volumes[i - 1] for i in range(1, self._window + 1)) != self._window
            ):
                continue

            signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest