from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves signal strong market sentiment. Large volumes "
        "accompanying price movements often indicate a significant shift in investor "
        "attitudes, leading to potentially profitable trading opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Identify strong price moves with high volume
        volumes = history["volume"].to_list()
        returns = [float(r) for r in history["r"].to_list()]

        signals: list[str] = []
        for symbol, vol, ret in zip(history.columns[:-2], volumes[1:], returns[1:]):
            if abs(ret) > 0.05 and vol >= max(volumes[-self._window :]) * 1.5:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest