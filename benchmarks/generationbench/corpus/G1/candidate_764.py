from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "By identifying symbols with significant price movements and corresponding volume increases, "
        "we can capitalize on momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            df = history.select(
                pl.col("session_date"),
                pl.col(symbol).alias("close"),
                (pl.col(symbol) / pl.col(symbol).shift(1) - 1.0).alias("r"),
                (pl.col(f"{symbol}_volume") / pl.col(f"{symbol}_volume").shift(1)).alias(
                    "v"
                ),
            )
            df = df.filter(pl.col("session_date") < view.as_of)
            if not df.height:
                continue
            last_session = df.sort("session_date", descending=True).select(
                ["close", "r", "v"]
            ).row(0)
            close_val, rel_change, vol_change = [float(v) for v in last_session]
            if abs(rel_change) >= 0.05 and vol_change > 1.2:
                picks.append(symbol)

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