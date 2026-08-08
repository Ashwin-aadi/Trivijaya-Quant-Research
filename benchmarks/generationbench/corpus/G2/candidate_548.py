from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate strong trends. A significant increase "
        "in volume on a price move suggests that the market is willing to put capital behind "
        "that direction, potentially leading to continuation of the trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_change_signal = (history["volume"] / history["volume"].shift(1) - 1.0).alias("vol_chg")
        price_move_signal = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("price_move")
        
        combined_signal = (
            history.with_columns(volume_change_signal, price_move_signal)
                   .filter((pl.col("vol_chg") > 0.25) & (pl.col("price_move") > 0.01))
                   .sort("session_date", descending=True)
                   .group_by("symbol")
                   .agg(pl.count().alias("count"))
                   .filter(pl.col("count") >= self._window / 2)
        )

        if combined_signal.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in combined_signal.to_dicts()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest