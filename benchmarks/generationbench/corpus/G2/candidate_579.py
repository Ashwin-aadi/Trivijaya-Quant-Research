from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "Assets with higher liquidity are typically more attractive for investors due to their "
        "lower transaction costs and reduced price impact. By equal-weighting assets based on their "
        "liquidity measures, we aim to capture the benefits of liquidity while diversifying across "
        "the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_weights = _compute_liquidity_weights(closes)

        total_weight = sum(liquidity_weights.values())
        normalized_weights = {s: w / total_weight for s, w in liquidity_weights.items()}

        return Signal(
            information_available_at=stamp,
            weights={s: normalized_weights[s] for s in view.symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_liquidity_weights(closes: pl.DataFrame) -> dict[str, float]:
    liquidity_weights: dict[str, float] = {}
    for symbol in closes.columns[1:]:
        volume = _calculate_volume(closes[symbol])
        if volume > 0:
            liquidity_weights[symbol] = volume

    return liquidity_weights


def _calculate_volume(close_prices: pl.Series) -> float:
    daily_volumes = close_prices.to_list()[1:]  # Drop the first element which is NaN
    total_volume = sum([v for v in daily_volumes if not v.is_nan()])
    return total_volume / len(daily_volumes)