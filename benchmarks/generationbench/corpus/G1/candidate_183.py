from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume are "
        "considered. Equal weighting among these stocks promotes diversification and can "
        "reduce the impact of individual stock-specific risks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .head(self._window)["symbol"]
        )

        if not liquidity_screened.to_list():
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(liquidity_screened)
        signal_weights = {symbol: equal_weight for symbol in liquidity_screened}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_pydatetime().date()
    assert isinstance(newest, date)
    return newest