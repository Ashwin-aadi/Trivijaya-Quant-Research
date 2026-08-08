from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment. "
        "A significant volume increase on a price move indicates potential continuation of the trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.select("symbol", "session_date").to_pandas().values.tolist():
                continue
            history_df = history.filter((pl.col("symbol") == symbol))
            recent_closes = [float(v) for v in history_df["close"].to_list()]
            recent_volumes = [int(v) for v in history_df["volume"].to_list()]

            if len(recent_closes) < self._window + 1 or len(recent_volumes) < self._window + 1:
                continue

            # Calculate the price change and volume change
            price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            volume_change = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]

            if price_change > 0.05 and volume_change > 0.3:
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
    newest = visible.select("session_date").max().cast(pl.Date).to_numpy()[0].item().date()
    assert isinstance(newest, date)
    return newest