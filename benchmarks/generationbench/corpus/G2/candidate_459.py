from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are typically more efficient in price discovery and less prone "
        "to abnormal price movements. By equal-weighting these liquid stocks, we can benefit "
        "from their stability and reduce the risk of holding illiquid, potentially volatile "
        "stocks."
    )

    def __init__(self, min_volume: int = 100000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        liquidity_filtered = history.filter(
            pl.col("volume") >= self._min_volume
        ).select(["symbol", "adj_close"]).group_by("symbol").agg(
            pl.col("adj_close").mean().alias("avg_price")
        )

        if liquidity_filtered.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest