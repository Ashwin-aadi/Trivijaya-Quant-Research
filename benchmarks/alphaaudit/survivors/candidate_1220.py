from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can "
        "potentially lead to sustained price movements. By identifying such moves, we aim "
        "to capture profitable trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            if df.height < self._window:
                continue

            prices = [float(v) for v in df["adj_close"].drop_nulls().to_list()]
            volumes = [int(v) for v in df["volume"].drop_nulls().to_list()]

            upmove, downmove = False, False
            last_trend = None
            for i in range(1, self._window):
                if prices[i] > prices[i - 1]:
                    trend = "up"
                elif prices[i] < prices[i - 1]:
                    trend = "down"
                else:
                    continue

                if (last_trend is not None and last_trend == "up" and
                        volumes[i] > volumes[i - 1]):
                    upmove = True
                elif (last_trend is not None and last_trend == "down" and
                      volumes[i] < volumes[i - 1]):
                    downmove = True

                if upmove and downmove:
                    break

                last_trend = trend

            if upmove:
                picks.append(symbol)

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