from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment. "
        "By identifying symbols that have both significant price movements and substantial volume, "
        "we can capitalize on periods of high momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            row = history.filter(pl.col("symbol") == symbol).sort(by="session_date").rows()
            opens, closes, highs, lows, volumes = (
                [float(r[1]) for r in row],
                [float(r[3]) for r in row],
                [float(r[2]) for r in row],
                [float(r[4]) for r in row],
                [float(r[6]) for r in row],
            )
            if len(opens) < self._window:
                continue

            close_change = (closes[-1] - opens[0]) / opens[0]
            volume_change = volumes[-1] - volumes[0]

            # Filter on significant price change and volume increase
            if abs(close_change) > 0.05 and volume_change > 10000:
                picks.append(symbol)

        picks = picks[:3]  # Limit to top 3 symbols
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest