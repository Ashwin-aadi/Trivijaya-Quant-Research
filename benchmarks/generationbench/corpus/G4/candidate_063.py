from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the anomaly in equity markets where smaller-cap stocks often exhibit higher returns due to lower liquidity. By screening for liquid stocks and employing an equal weighting approach, we balance exposure across a diverse set of companies while leveraging potential outperformance from less-liquidity-constrained firms."
    )

    def __init__(self, window: int = 20, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volume_df = (
            history.select("symbol", pl.col("volume"))
            .group_by("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .sort("avg_volume", descending=True)
        )

        if volume_df.height < self._top_n:
            top_symbols = [row["symbol"] for row in volume_df.rows()]
        else:
            top_symbols = [row["symbol"] for row in volume_df.head(self._top_n).rows()]

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