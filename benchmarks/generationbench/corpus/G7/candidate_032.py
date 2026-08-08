from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy identifies stocks with volume-confirmed directional moves. "
        "By focusing on sustained upward trends supported by increased trading activity, we aim to capitalize on momentum."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            opens = [float(v) for v in df["open"].to_list()]
            closes = [float(v) for v in df["close"].to_list()]
            volumes = [int(v) for v in df["volume"].to_list()]

            if len(opens) < self._window:
                continue

            up_days = sum(1 for i in range(self._window - 1) if closes[i + 1] > opens[i])
            avg_volume = float(df.select(pl.col("volume").mean()).item())
            volume_up_days = sum(1 for v in volumes[-self._window:] if v > avg_volume)

            if up_days >= self._top_n and volume_up_days >= self._top_n:
                picks.append(symbol)

        picks = picks[: self._top_n]
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
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest