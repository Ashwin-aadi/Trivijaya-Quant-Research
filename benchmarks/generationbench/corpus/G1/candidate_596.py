from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can lead to "
        "potentially profitable trades. By identifying symbols with significant volume increases"
        " on a breakout day, we aim to capture these moves."
    )

    def __init__(self, window: int = 20, min_volume_increase: float = 1.5) -> None:
        self._window = window
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.select(["session_date", "symbol", "close", "volume"])
            close_series = df.filter(pl.col("symbol") == symbol).select("close")
            volume_series = df.filter(pl.col("symbol") == symbol).select("volume")

            if not (close_series.height > self._window and volume_series.height > self._window):
                continue

            closes = [float(v) for v in close_series.drop_nulls().to_list()]
            volumes = [int(v) for v in volume_series.drop_nulls().to_list()]

            breakout_day = max(range(self._window - 1, len(closes)), key=lambda i: closes[i])
            if (
                breaks_out(closes, self._window)
                and volumes[breakout_day] > min_volume_increase * max(volumes)
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


def breaks_out(closes: list[float], window: int) -> bool:
    for i in range(window - 1, len(closes)):
        if closes[i] > max(closes[:i]):
            return True
    return False