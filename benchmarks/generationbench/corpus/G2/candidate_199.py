from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression is a situation where volatility has decreased significantly, "
        "potentially signaling an impending breakout. A high concentration of trading in "
        "narrow price ranges can lead to increased momentum when the market breaks out."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the range for each symbol
        history = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(
                [
                    pl.col("close").first().alias("start_price"),
                    pl.col("range").mean().alias("avg_range"),
                ]
            )
        )

        # Calculate the compressed range
        history = (
            history.with_columns(
                (pl.col("range") / pl.col("avg_range")).alias("compression_factor")
            )
            .sort("compression_factor", descending=True)
            .with_column(pl.arange(0, history.height).alias("row_number"))
            .filter(
                (pl.col("row_number") >= history.height - 5) & (pl.col("row_number") < history.height)
            )
        )

        # Pick symbols with the highest compression factor
        picks = [str(row["symbol"]) for row in history.to_dicts()]

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