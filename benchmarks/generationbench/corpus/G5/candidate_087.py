from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for highly liquid stocks by selecting those with a daily "
        "trading volume above the market median. The selected assets are then equal-weighted."
    )

    def __init__(self, window: int = 20, liquidity_threshold_percentile: float = 75) -> None:
        self._window = window
        self._liquidity_threshold_percentile = liquidity_threshold_percentile

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_col = "volume"
        median_volume = float(history[volume_col].quantile(self._liquidity_threshold_percentile / 100))
        liquidity_screened_history = (
            history.filter(pl.col(volume_col) > median_volume)
                   .group_by(["symbol"])
                   .agg((pl.col(volume_col).mean().alias("avg_volume")))
        )
        
        if liquidity_screened_history.height < 5:
            return Signal(information_available_at=stamp, weights={})

        symbols = [row[0] for row in liquidity_screened_history.to_dicts()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest