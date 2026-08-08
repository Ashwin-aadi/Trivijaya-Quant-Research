from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy identifies significant price movements confirmed by high trading volumes. "
        "By combining daily OHLC data with volume thresholds and percentage changes, it ensures robust signals for entering long positions."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            adj_closes = hist.select(["adj_close", "volume"])
            if adj_closes.height < self._window:
                continue

            # Calculate daily returns and relative volume
            adj_closes = (
                adj_closes.with_columns(
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
                )
                .with_columns(
                    (pl.col("volume") / pl.col("volume").shift(1)).alias("relative_volume")
                )
                .drop_nulls()
            )

            # Check for directional moves with high volume confirmation
            last_close = float(adj_closes["adj_close"][-1])
            if adj_closes["r"][-1] > 0.05 and adj_closes["volume"][-1] >= adj_closes["relative_volume"].mean() * 2:
                picks.append(symbol)
            elif adj_closes["r"][-1] < -0.05 and adj_closes["volume"][-1] >= adj_closes["relative_volume"].mean() * 2:
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
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest