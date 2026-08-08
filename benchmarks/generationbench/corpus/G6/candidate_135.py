from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy selects the top 50 most liquid stocks based on their average daily trading "
        "volume over the past month and weights them equally. It ensures balanced risk exposure "
        "and simplicity while maintaining sufficient liquidity."
    )

    def __init__(self, window: int = 30, min_volume: float = 10e6, top_n: int = 50) -> None:
        self._window = window
        self._min_volume = min_volume
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_df = history.select(
            pl.col("symbol"),
            pl.col("volume").mean().alias("avg_volume"),
        ).sort("avg_volume", descending=True).head(self._top_n)

        eligible_symbols = volume_df.filter(pl.col("avg_volume") >= self._min_volume)[
            "symbol"
        ].to_list()

        if len(eligible_symbols) < 2:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(eligible_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in eligible_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest