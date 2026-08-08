from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on small-cap stocks with high liquidity. By equal-weighting "
        "these stocks, we aim to capture higher returns from under-followed firms while managing risk."
    )

    def __init__(self, window: int = 180, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
                   .agg((pl.col("volume").mean()).alias("avg_volume"))
                   .sort("avg_volume", descending=True)
        )

        small_caps = view.closes(lookback=self._window).select(
            [pl.col(sym) for sym in liquidity_screened.select("symbol") if
             float(view.latest_close()[sym]) < 100]  # Assuming a simple cap threshold
        ).shape[1]

        if small_caps < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row['symbol'] for row in liquidity_screened.to_dicts()[:self._top_n]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest