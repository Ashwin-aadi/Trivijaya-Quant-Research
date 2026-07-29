from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment and can "
        "indicate potential continuation of the trend. By identifying such moves, we aim to "
        "capitalize on the momentum."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily percentage change in price
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )

        # Identify directional moves with high volume
        moves = (
            history.filter(
                (pl.col("volume") > pl.col("volume").quantile(0.75))
                & ((pl.col("r") > 0.01) | (pl.col("r") < -0.01))
            )
        ).select(["symbol", "session_date"])

        if moves.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols that do not show a consistent directional move
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in moves["symbol"].to_list():
                continue
            sessions = history.filter(pl.col("symbol") == symbol).select(
                "session_date", "r"
            )
            recent_sessions = (
                sessions.sort("session_date").tail(3)
            )  # Look at the last 3 days to confirm consistency
            if all(recent_sessions["r"].to_list()) > 0.01 or all(
                recent_sessions["r"].to_list()
            ) < -0.01:
                picks.append(symbol)

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