from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment. "
        "A significant increase in volume on a price move suggests that the move is not just "
        "a short-term fluctuation but rather driven by substantial trading activity, which can "
        "indicate a trend continuation or reversal."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            if hist.is_empty():
                continue

            latest_close = float(hist.select(pl.last("adj_close"))[0][0])
            prev_close = float(hist.select(pl.last("close"))[0][0])

            # Calculate the percentage change
            price_change = (latest_close - prev_close) / prev_close * 100.0

            # Get volume data for the last day and compare with previous session's volume
            latest_volume = float(hist.select(pl.last("volume"))[0][0])
            prev_volume = float(hist.filter(pl.col("session_date") == (view.as_of - date(1, 1, 1))).select(pl.first("volume")))[0][0]

            # Volume change percentage
            volume_change = (latest_volume - prev_volume) / prev_volume * 100.0 if prev_volume != 0 else 0

            # Check for a significant price move with high volume confirmation
            if abs(price_change) > 2.0 and abs(volume_change) > 30.0:
                picks.append(symbol)

        picks = picks[:5]  # Select top 5 symbols only
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest