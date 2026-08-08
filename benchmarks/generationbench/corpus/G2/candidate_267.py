from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more likely to be continuously priced and have less "
        "idiosyncratic risk. By equal-weighting these liquid assets, we can capture the "
        "average performance of the most traded stocks in the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = _calculate_liquidity_score(history)
        weights = _equal_weighting_with_liquidity(liquidity_scores)

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol, weight in zip(view.symbols, weights)
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_liquidity_score(history: pl.DataFrame) -> pl.Series:
    volume = history.select(pl.col("volume").sum()).to_series()
    average_volume = volume / len(view.symbols)

    liquidity_scores = (
        history.select(
            (pl.col("volume") - pl.col("volume").shift(1)).abs().mean().alias("avg_abs_change")
        )
        .with_column((pl.col("avg_abs_change") / average_volume).rank(method="ordinal", descending=True))
        .select(pl.col("symbol"), "r")
    )

    return liquidity_scores.explode("r")


def _equal_weighting_with_liquidity(liquidity_scores: pl.DataFrame) -> list[float]:
    symbols = [row[0] for row in liquidity_scores.to_list()]
    scores = [row[1] for row in liquidity_scores.to_list()]

    total_score = sum(scores)
    if total_score == 0:
        return [1.0 / len(symbols)] * len(symbols)

    weights = [(score / total_score) for score in scores]
    normalized_weights = [weight / max(weights) for weight in weights]

    return normalized_weights