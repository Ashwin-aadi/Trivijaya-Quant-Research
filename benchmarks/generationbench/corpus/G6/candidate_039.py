from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidEqualWeighting(Strategy):
    rationale = (
        "By focusing on highly liquid stocks and implementing equal weighting, this strategy "
        "aims to balance diversification with risk management in the Indian market."
    )

    def __init__(self, min_volume: int = 50_000_000, lookback: int = 30, max_names: int = 100) -> None:
        self._min_volume = min_volume
        self._lookback = lookback
        self._max_names = max_names

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter stocks based on minimum volume
        filtered = (
            history.filter(
                (pl.col("volume") > self._min_volume)
                & (pl.col("symbol").is_in(view.symbols))
            )
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg_close"))
            .sort("avg_close", descending=True)
            .head(self._max_names)
        )

        if filtered.height < 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in filtered.rows()]

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