from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are more likely to be sustainable and "
        "profitable. This strategy identifies symbols that show a significant increase in "
        "volume alongside a strong price movement."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_threshold = 1.25 * history.select(pl.col("volume").mean()).item()
        price_move_threshold = 0.02

        symbols_of_interest: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            daily_changes = (
                history.filter(pl.col("symbol") == symbol)
                .select(["session_date", "open", "close"])
                .with_columns(
                    (pl.col("close") - pl.col("open")) / pl.col("open").shift(1) - 1.0
                ).sort("session_date")
            )
            if daily_changes.is_empty():
                continue

            latest_change = daily_changes.select("change").item()
            volume_change = (
                history.filter(pl.col("symbol") == symbol)
                .select("volume")
                .with_columns(
                    (pl.col("volume") - pl.col("volume").shift(1)) / pl.col("volume").shift(1) * 100
                )
                .sort("session_date")
                .select("change_volume_percent")
                .item()
            )

            if (
                latest_change > price_move_threshold
                and volume_change > volume_threshold
            ):
                symbols_of_interest.append(symbol)

        if not symbols_of_interest:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_of_interest)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_of_interest},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest