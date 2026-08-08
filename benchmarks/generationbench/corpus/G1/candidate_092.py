from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can "
        "suggest continuation of the trend. High volume on a breakout or breakdown is often"
        " indicative of significant investor interest."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in history.select("symbol").column_names:
                continue

            session_dates = [float(v) for v in history["session_date"].to_list()]
            opens = [float(v) for v in history[symbol]["open"].to_list()]
            closes = [float(v) for v in history[symbol]["close"].to_list()]
            volumes = [float(v) for v in history[symbol]["volume"].to_list()]

            last_close = closes[-1]
            if (last_close - opens[0]) * (closes.index(last_close) - opens.index(opens[0])) > 0:
                # Directional move check
                volume_on_breakout = volumes[closes.index(last_close)]
                avg_volume = sum(volumes) / len(volumes)
                if volume_on_breakout > 1.5 * avg_volume:
                    picks.append(symbol)

        picks = picks[:3]
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