from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of significant market "
        "sentiment shifts. By identifying stocks that have experienced a large volume spike on "
        "a price breakout, we can potentially capture the momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            if symbol_history.is_empty():
                continue

            # Calculate daily price change and volume
            symbol_history = symbol_history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                (pl.col("volume") / pl.col("volume").shift(1)).alias("volume_ratio"),
            )

            # Find the day with the largest return
            max_return_day = symbol_history.sort("return", descending=True).head(1)

            if not max_return_day.is_empty():
                breakout_price = max_return_day.select("adj_close")[0, 0]
                breakout_volume = max_return_day.select("volume")[0, 0]

                # Check for volume confirmation
                post_breakout_history = symbol_history.filter(
                    (pl.col("adj_close") > breakout_price) & (pl.col("session_date") > max_return_day["session_date"][0])
                )
                if not post_breakout_history.is_empty():
                    avg_volume_after_breakout = post_breakout_history.select(pl.col("volume").mean())[0, 0]
                    if avg_volume_after_breakout >= self._threshold * breakout_volume:
                        breakout_symbols.append(symbol)

        # Select top N symbols
        breakout_symbols = breakout_symbols[:5]

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest