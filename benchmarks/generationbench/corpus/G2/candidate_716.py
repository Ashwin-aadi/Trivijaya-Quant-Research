from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong conviction from market "
        "participants. If a stock's price moves significantly and is accompanied by increased "
        "volume, it suggests that the move is driven by real buying or selling pressure rather "
        "than noise. Capturing such moves can lead to profitable trades."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            hist = history.filter(pl.col("symbol") == symbol)
            latest_close = float(view.latest_close()[symbol])
            open_price = float(hist.select(pl.col("open").last()).item())
            high = float(hist.select(pl.col("high").max()).item())
            low = float(hist.select(pl.col("low").min()).item())
            volume_change = float(
                (latest_close - open_price) / open_price
            ) * hist.select(pl.col("volume").sum()).item()
            if latest_close > high:
                picks.append(symbol)
            elif latest_close < low:
                picks.append(symbol)

        weights = {s: 1.0 for s in picks} if picks else {}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest