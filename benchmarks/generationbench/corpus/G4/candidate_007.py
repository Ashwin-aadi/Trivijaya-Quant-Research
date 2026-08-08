from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionAndCompression(Strategy):
    rationale = (
        "This strategy exploits market conditions where stocks exhibit high volatility or are trading within tight ranges. "
        "High dispersion indicates potential arbitrage or mean reversion opportunities, while range compression suggests breakout scenarios."
    )

    def __init__(self, atr_window: int = 14, sd_window: int = 20, max_positions: int = 20) -> None:
        self._atr_window = atr_window
        self._sd_window = sd_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._atr_window + self._sd_window)

        if history.height < self._atr_window + self._sd_window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate ATR for dispersion
        atr_data = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .select(
                pl.col("session_date"),
                "symbol",
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("high").shift(-1) - pl.col("low").shift(-1)).abs().alias("prev_range")
            )
            .group_by("symbol", maintain_order=True)
            .agg(
                (
                    (pl.col("range") + pl.col("prev_range")) / 2.0,
                    pl.col("session_date").sort(descending=False).first(),
                ).alias("atr"),
                pl.count().alias("count")
            )
        )

        atr_values = (
            atr_data.with_columns(
                (pl.col("atr") / pl.col("atr").shift(self._atr_window - 1) - 1.0).alias("atr_growth_rate")
            )
            .filter(pl.col("count") >= self._atr_window)
            .sort("atr_growth_rate", descending=True)
            .head(self._max_positions)
            .select(["symbol"])
        )["symbol"].to_list()

        # Calculate Standard Deviation for range compression
        sd_data = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .select(
                pl.col("session_date"),
                "symbol",
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol", maintain_order=True)
            .agg(
                (
                    pl.col("range").std().alias("sd"),
                    pl.count().alias("count")
                )
            )
        )

        sd_values = (
            sd_data.with_columns(
                (pl.col("sd") / pl.col("sd").shift(self._sd_window - 1) - 1.0).alias("sd_growth_rate")
            )
            .filter(pl.col("count") >= self._sd_window)
            .sort("sd_growth_rate")
            .head(self._max_positions)
            .select(["symbol"])
        )["symbol"].to_list()

        # Combine results
        combined_symbols = set(atr_values) | set(sd_values)
        if not combined_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in combined_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest