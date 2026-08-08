from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedMomentumStrategy(Strategy):
    rationale = (
        "This strategy selects the top 30 stocks by cross-sectional momentum, "
        "considering both price and volume. It exits when momentum weakens, falls below moving averages, or fails to hit new highs."
    )

    def __init__(self, window_price: int = 20, window_volume: int = 100, top_n: int = 30) -> None:
        self._window_price = window_price
        self._window_volume = window_volume
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_price + self._window_volume)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        price_momentum = (
            history.group_by("symbol")
            .agg(
                pl.col("adj_close").rolling_mean(self._window_price).alias(f"rma_{self._window_price}"),
                pl.col("adj_close").rolling_mean(self._window_volume).alias(f"rma_{self._window_volume}"),
            )
            .with_columns(
                (pl.col(f"rma_{self._window_price}") - pl.col(f"rma_{self._window_volume}")).alias("price_momentum")
            )
        )

        volume_momentum = (
            history.group_by("symbol")
            .agg(pl.col("volume").rolling_mean(self._window_price).alias(f"vma_{self._window_price}"))
            .with_columns(
                (pl.col(f"vma_{self._window_price}") / pl.col("volume") * 100).alias("volume_ratio")
            )
        )

        combined = price_momentum.join(volume_momentum, on="symbol", how="inner")

        if combined.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        combined = (
            combined.sort("price_momentum", descending=True)
            .head(self._top_n)
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
        )

        top_symbols = [symbol for symbol in combined["symbol"].to_list()]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest