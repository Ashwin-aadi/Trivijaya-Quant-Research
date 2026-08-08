from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalVolume(Strategy):
    rationale = (
        "Historical data suggests that trading volumes exhibit seasonal patterns. By analyzing "
        "the average daily volume over a 3-year period, we can identify months with higher liquidity "
        "and investor interest. This allows us to time our trades for periods of increased market activity."
    )

    def __init__(self, window: int = 1095, top_n: int = 10, stop_loss: float = -0.05) -> None:
        self._window = window
        self._top_n = top_n
        self._stop_loss = stop_loss

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_avg = (
            history.group_by("symbol")
                   .agg((pl.col("volume").mean().alias("avg_volume")))
                   .sort("avg_volume", descending=True)
                   .select("symbol", "avg_volume")
                   .to_pandas()
        )

        symbols = list(volume_avg["symbol"].head(self._top_n))
        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(
            information_available_at=stamp, 
            weights={s: w for s, w in weights.items() if view.latest_close().get(s, 0) * (1 + self._stop_loss) > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest