from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendAndVolume(Strategy):
    rationale = (
        "This strategy seeks to identify stocks with a strong upward trend while simultaneously "
        "having high trading volume. A combination of both indicators could suggest that the stock is "
        "attracting significant interest from investors and may continue its upward trajectory."
    )

    def __init__(self, window: int = 20, min_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._min_volume_ratio = min_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the percentage change in price over the lookback period
        price_change = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("price_change")
        history = history.with_columns(price_change)

        # Filter symbols with high price change and sufficient volume
        strong_trend_symbols = (
            history.filter(
                (pl.col("price_change") > 0.05)
                & (history["volume"] / pl.col("volume").shift(1) > self._min_volume_ratio)
            )
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("m"))
        )

        if strong_trend_symbols.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in strong_trend_symbols.sort("m", descending=True).rows()]

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