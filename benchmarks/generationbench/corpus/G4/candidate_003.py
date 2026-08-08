from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where liquid stocks often exhibit higher returns "
        "due to lower transaction costs and better market coverage. By screening for liquidity and "
        "applying an equal weighting approach to these screened stocks, we aim to capture the benefits "
        "of higher liquidity while maintaining a balanced portfolio distribution."
    )

    def __init__(self, window: int = 20, max_positions: int = 100) -> None:
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_series = (
            history.select(pl.col("symbol"), pl.col("volume"))
            .group_by("symbol")
            .agg((pl.col("volume").mean().alias("avg_volume")))
            .sort("avg_volume", descending=True)
            .head(self._max_positions)["symbol"]
            .to_list()
        )

        if not volume_series:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volume_series)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volume_series},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest