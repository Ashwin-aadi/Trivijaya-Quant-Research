from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy identifies stocks showing significant price movements confirmed by "
        "high trading volumes to indicate strong buying or selling pressure. It combines "
        "volume and directional signals for robust trend identification and incorporates moving"
        " averages for precise entry/exit timing."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate 20-day simple moving average (SMA)
        sma_20 = history.select(
            pl.col("close").rolling_mean(self._window).alias(f"sma_{self._window}")
        )

        # Filter for significant price movements and high volume
        filtered_history = (
            history.join(sma_20, on="session_date")
            .with_columns(
                (pl.col("close") / pl.col(f"sma_{self._window}") - 1.0).alias("price_move_ratio"),
                ((pl.col("volume") > view.volume_percentile(90)).cast(pl.int8)).alias("high_volume")
            )
            .filter(
                (pl.col("price_move_ratio").abs() >= 0.02)
                & (pl.col("high_volume"))
            )
        )

        if filtered_history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            symbol_df = filtered_history.filter(pl.col("symbol") == symbol)
            sma_20_values = [float(v) for v in sma_20[symbol].to_list()]
            last_sma_20 = sma_20_values[-1]

            # Check if the closing price is above or below SMA and volume conditions
            long_condition = (
                (symbol_df["close"] > last_sma_20) &
                (symbol_df["volume"] > view.volume_percentile(90))
            )
            short_condition = (
                (symbol_df["close"] < last_sma_20) &
                (symbol_df["volume"] > view.volume_percentile(90))
            )

            if long_condition.sum() >= 3:
                picks.append(symbol)
            elif short_condition.sum() >= 3:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest