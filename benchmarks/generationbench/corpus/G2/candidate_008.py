from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can lead to "
        "sustained price trends. By identifying stocks with significant volume surges on their "
        "price increases or decreases, we can capitalize on the momentum."
    )

    def __init__(self, lookback_days: int = 30, min_volume_increase: float = 2.0) -> None:
        self._lookback_days = lookback_days
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = (
            history.group_by("symbol")
            .agg(
                pl.col("adj_close").first().alias("open"),
                pl.col("adj_close").last().alias("close"),
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("volume_ratio"),
            )
            .sort(
                "symbol",
                pl.col("volume_ratio").rank(method="ordinal", descending=True),
            )
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in symbol_data.symbol.to_list():
                continue

            row = symbol_data.filter(pl.col("symbol") == symbol).rows()[0]
            open_price, close_price, total_volume, volume_ratio = (
                float(row["open"]),
                float(row["close"]),
                float(row["total_volume"]),
                float(row["volume_ratio"]),
            )

            if volume_ratio > self._min_volume_increase:
                if close_price > open_price and (volume_ratio - 1) > 0.5 * self._min_volume_increase:
                    picks.append(symbol)
                elif close_price < open_price and (volume_ratio - 1) > 0.5 * self._min_volume_increase:
                    picks.append(symbol)

        picks = list(set(picks))[:5]  # Avoid duplicates and limit to top 5
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