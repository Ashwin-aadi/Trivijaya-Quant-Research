from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reverts to the mean over time. By using a trailing reference level, we can "
        "identify overbought or oversold conditions and generate trade signals accordingly."
    )

    def __init__(self, window: int = 50, std_dev: float = 2.0) -> None:
        self._window = window
        self._std_dev = std_dev

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_price = (
            closes
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).abs()
                .rank(method="dense", descending=False).alias("distance_from_mean")
            )
        )

        # Calculate the trailing reference level
        ref_level = (
            closes
            .group_by("symbol")
            .agg(
                pl.col("adj_close").shift(self._window // 2)
                .mean().alias("ref_level")
            )
        )

        merged = mean_price.join(ref_level, on="symbol", how="inner")

        # Generate signals for symbols with a large distance from the trailing reference
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in merged.columns:
                continue
            values = [float(v) for v in merged[merged["symbol"] == symbol]["distance_from_mean"].to_list()]
            if len(values) < self._window // 2 + 1:
                continue
            if max(values) > self._std_dev:
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest