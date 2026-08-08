from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting allocates capital to assets based on their "
        "trading volume. This approach aims to leverage the most liquid stocks in the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_sum = (
            history.group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
        )

        total_volume = float(volume_sum.select(pl.col("total_volume")).sum().item())
        if total_volume == 0:
            return Signal(information_available_at=stamp, weights={})

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            row = volume_sum.filter(pl.col("symbol") == symbol).to_dict(as_series=False)
            if not row or "total_volume" not in row:
                continue
            weight = float(row["total_volume"][0]) / total_volume
            picks[symbol] = weight

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in sorted(picks.items(), key=lambda x: -x[1])},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest