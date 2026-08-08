from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong momentum in a security. "
        "By identifying such moves, we can capitalize on the likely continuation of trends."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            # Get the last return value
            latest_return = float(history[symbol]["return"].last())

            # Check if it's a significant move (positive or negative)
            if abs(latest_return) > 0.01:  # Threshold for significant move
                # Confirm with volume
                volume_series = history.select(pl.col(symbol).filter(pl.col("session_date") < pl.lit(stamp))).select("volume")
                mean_volume = float(volume_series["volume"].mean())
                if latest_return > 0 and history[symbol]["volume"].last() > 1.5 * mean_volume:
                    picks.append(symbol)
                elif latest_return < 0 and history[symbol]["volume"].last() > 1.5 * mean_volume:
                    picks.append(symbol)

        picks = list(set(picks))[:5]  # Remove duplicates and limit to top 5
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest