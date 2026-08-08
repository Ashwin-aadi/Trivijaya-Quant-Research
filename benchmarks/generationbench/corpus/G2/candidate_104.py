from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment. "
        "When a stock's price moves in the direction of its recent trend and is accompanied by high volume, "
        "it suggests that institutional or retail investors are actively participating in this move. "
        "Such moves often precede further price appreciation."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the price change over the window period
        price_change = (history["adj_close"].shift(-1) - history["adj_close"]).alias("price_change")
        history = history.with_columns(price_change)

        # Filter for stocks that are moving in a direction consistent with their recent trend
        history = history.with_column(
            pl.when(history["close"].shift(1) < history["open"].shift(1)).then(-1).otherwise(1).alias("trend")
        )
        history = history.with_column(
            (history["price_change"] * history["trend"]).alias("directional_move")
        )

        # Ensure the directional move is positive and volume is high
        high_volume_mask = history["volume"].gt(history["volume"].mean())
        positive_directional_move_mask = history["directional_move"].gt(0)
        filtered_history = history.select(
            pl.all().filter(high_volume_mask & positive_directional_move_mask)
        )

        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols_with_positive_moves = filtered_history["symbol"].to_list()
        weight_per_symbol = 1.0 / len(symbols_with_positive_moves)

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol for symbol in symbols_with_positive_moves
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest