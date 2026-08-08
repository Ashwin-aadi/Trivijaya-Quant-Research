from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompression(Strategy):
    rationale = (
        "This strategy exploits dispersion and range compression in the Indian equity market by "
        "rotating sectors based on their performance. During dispersion phases, it identifies the "
        "outperforming sector. During range compression, it diversifies to mitigate risks."
    )

    def __init__(self, window: int = 90, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns for each sector
        history = (
            history
            .with_columns(
                (pl.col("close") / pl.col("open").shift(1) - 1.0).alias("daily_return")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(
                pl.col("daily_return").mean().alias("avg_return"),
                pl.col("daily_return").stddev().alias("std_dev"),
                (pl.col("high") - pl.col("low")).abs().mean().alias("range_compression"),
            )
        )

        # Rank sectors based on dispersion and range compression
        dispersion = history.sort("std_dev", descending=True).head(self._top_n)
        compression = (
            history.sort("range_compression", ascending=True).head(self._top_n)
        )

        if not dispersion.height:
            return Signal(information_available_at=stamp, weights={})

        # Determine the phase and allocate accordingly
        current_phase = "dispersion" if dispersion.height > 1 else "compression"
        sectors = (
            [sector.item() for sector in dispersion["symbol"].to_list()]
            + [sector.item() for sector in compression["symbol"].to_list()]
        )

        weights = {s: 0.2 / len(sectors) for s in sectors}
        if current_phase == "dispersion":
            weights[dispersion["symbol"][0].item()] += 0.6

        return Signal(
            information_available_at=stamp,
            weights={s: float(weights[s]) for s in sectors},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest