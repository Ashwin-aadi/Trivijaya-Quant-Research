from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedTrend(Strategy):
    rationale = (
        "This strategy exploits significant price movements that are volume-confirmed. "
        "Identifying such trends helps in capitalizing on the consistency between price and "
        "volume during strong market moves, providing robust trading opportunities."
    )

    def __init__(self, min_price_change: float = 0.02, lookback_volume_days: int = 20) -> None:
        self._min_price_change = min_price_change
        self._lookback_volume_days = lookback_volume_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_volume_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_avg_by_symbol = (
            history.group_by("symbol")
                   .agg((pl.col("volume") / pl.col("volume").mean().alias("volume_ratio"))
                        .filter(pl.col("volume_ratio") > 1.0)
                        .count()
                        .alias("vol_confirmed_days"),
                        (pl.col("adj_close") - pl.col("adj_close").shift(1)) / pl.col("adj_close").shift(1) * 100
                        .filter(pl.col("*") >= self._min_price_change)
                        .count()
                        .alias("price_move_days"))
        )

        filtered_symbols = volume_avg_by_symbol.filter(
            (pl.col("vol_confirmed_days") > 0) & (pl.col("price_move_days") > 0)
        ).select("symbol").to_list()

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest