from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy identifies stocks showing a directional price move accompanied by increasing volume. "
        "Large institutional or retail investors often initiate significant trades, pushing stock prices in one direction while generating substantial trading volumes. "
        "The approach leverages both directional momentum and volume confirmation for potential gains."
    )

    def __init__(self, window: int = 20, price_threshold: float = 0.01, volume_threshold: float = 1.5) -> None:
        self._window = window
        self._price_threshold = price_threshold
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()
        symbols = [symbol for symbol in view.symbols if symbol in latest_close.keys()]

        candidates: list[str] = []
        for symbol in symbols:
            price_changes = (
                history.filter(pl.col("symbol") == symbol)
                       .select((pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1).alias("price_change"))
            )
            volume_changes = (
                history.filter(pl.col("symbol") == symbol)
                       .select(
                           (pl.col("volume") / latest_close[symbol] * 100).alias("volume_ratio"),
                           (pl.col("volume").rolling_mean(self._window)).alias("avg_volume")
                       ).with_columns((pl.col("volume") / pl.col("avg_volume") - 1).alias("vol_change"))
            )

            if price_changes.height < self._window:
                continue

            last_price_change = float(price_changes.sort("session_date", descending=False)["price_change"].tail(1)[0])
            vol_change = float(volume_changes.sort("session_date", descending=False)["vol_change"].tail(1)[0])

            if (last_price_change > self._price_threshold and vol_change > self._volume_threshold - 1) or \
               (last_price_change < -self._price_threshold and vol_change > self._volume_threshold - 1):
                candidates.append(symbol)

        weights = {symbol: 1.0 / len(candidates) for symbol in candidates}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest