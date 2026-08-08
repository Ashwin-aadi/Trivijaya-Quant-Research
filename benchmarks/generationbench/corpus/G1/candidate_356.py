from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong buying or selling pressure. "
        "We exploit this by identifying symbols that show a significant change in direction "
        "with volume support."
    )

    def __init__(self, window: int = 10, min_volume_change: float = 50) -> None:
        self._window = window
        self._min_volume_change = min_volume_change

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily price change and volume
        history_with_change = (
            history.with_columns(
                (pl.col("close") - pl.col("adj_close").shift(1)).alias("price_change"),
                (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("vol_change"),
            )
            .sort("session_date")
            .fill_null(pl.lit(0))
        )

        # Identify symbols with significant price and volume changes
        significant_moves = []
        for symbol in view.symbols:
            if symbol not in history_with_change.columns:
                continue
            price_change = history_with_change[symbol]["price_change"].to_list()
            vol_change = history_with_change[symbol]["vol_change"].to_list()

            if len(price_change) < self._window or any(pl.col("vol_change").is_nan()):
                continue

            last_price_change = price_change[-1]
            last_vol_change = vol_change[-1]

            if abs(last_price_change) > 0.05 and abs(last_vol_change) > self._min_volume_change:
                significant_moves.append(symbol)

        # Limit the number of picks to top_n
        significant_moves = significant_moves[:3]  # Adjust as needed

        if not significant_moves:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(significant_moves)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in significant_moves},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest