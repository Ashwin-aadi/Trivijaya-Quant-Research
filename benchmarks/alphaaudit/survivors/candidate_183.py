from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume are "
        "considered for equal weighting. This strategy aims to reduce the risk associated "
        "with illiquid stocks and potentially improve portfolio diversification."
    )

    def __init__(self, min_volume: float = 1000000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average volume over 1 year
        avg_volume = history.group_by("symbol").agg(
            pl.col("volume").mean().alias("avg_volume")
        )
        selected_symbols = (
            avg_volume.filter(pl.col("avg_volume") > self._min_volume)
            .select("symbol")
            .to_pandas()["symbol"]
            .tolist()
        )

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight each selected symbol
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest