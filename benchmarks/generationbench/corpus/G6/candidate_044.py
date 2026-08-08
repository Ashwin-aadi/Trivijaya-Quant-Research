from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy enters positions based on volume-confirmed directional moves in the stock prices. "
        "It combines the high-volume entry criteria with trend confirmation through OHLC data to balance "
        "ambition and conservatism."
    )

    def __init__(self, lookback: int = 20, min_volume_increase: float = 1.5) -> None:
        self._lookback = lookback
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback + 3)
        if history.height < self._lookback + 3:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            data = (
                history
                .filter(pl.col("symbol") == symbol)
                .sort("session_date")
                .select(
                    pl.col("session_date"),
                    (pl.col("adj_close") / pl.col("open") - 1.0).alias("close_to_open_ratio"),
                    (pl.col("high") - pl.col("low")).alias("true_range"),
                    (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("volume_increase")
                )
            )

            last_close = data.select(pl.col("adj_close").last().alias("last_close")).to_dict(False)[0]["last_close"]
            if last_close > history.filter(pl.col("symbol") == symbol)["close_to_open_ratio"].last() and \
               data["volume_increase"].max() >= self._min_volume_increase:
                signals.append(symbol)

        signals = signals[:5]
        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest