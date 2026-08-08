from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and reliability. By focusing on the most "
        "liquid stocks, we aim to reduce transaction costs and ensure that our trades are "
        "practicable without significantly moving the market."
    )

    def __init__(self, min_trading_volume: float = 1000000) -> None:
        self._min_trading_volume = min_trading_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.height < 365:
            return Signal(information_available_at=stamp, weights={})

        # Filter by minimum trading volume
        filtered_history = history.filter(
            (pl.col("volume") > self._min_trading_volume).any().over("symbol")
        )

        # Calculate liquidity score as the reciprocal of average daily volume
        liquidity_scores = (
            filtered_history.group_by("symbol")
            .agg((1 / pl.col("volume").mean()).alias("liquidity_score"))
            .sort("liquidity_score", descending=True)
            .head(10)
        )

        # Convert to dictionary for weighting
        top_symbols = [row["symbol"] for row in liquidity_scores.to_dict(as_series=False).values()]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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