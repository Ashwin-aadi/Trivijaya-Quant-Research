from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedStrategy(Strategy):
    rationale = (
        "Liquidity can be a proxy for market interest in a stock. Higher liquidity may indicate "
        "greater investor confidence and lower transaction costs, potentially leading to more stable prices."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
                   .agg((pl.col("volume").mean() / pl.col("volume").std()).alias("liquidity_score"))
        )

        if liquidity_scores.height < 1:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = liquidity_scores.sort("liquidity_score", descending=True)["symbol"].to_list()
        total_liquidity = sum(float(v) for v in history[history["symbol"] == sorted_symbols[0]]["volume"].to_list())

        weights = {s: float(lv / total_liquidity) for s, lv in zip(sorted_symbols, history.select(pl.col("volume").sum().alias("total_volume"))["total_volume"].to_list())}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest