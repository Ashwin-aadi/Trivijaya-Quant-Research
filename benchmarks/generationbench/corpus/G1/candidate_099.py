from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity plays a crucial role in trading costs and the ability to quickly enter or exit "
        "positions. A liquidity-screened equal-weight strategy ensures that only highly liquid "
        "stocks are considered for investment, thereby minimizing these costs."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score")
            )
            .sort("liquidity_score", descending=True)
            .head(50)["symbol"]
        )

        if liquidity_scores.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: 1.0 / len(liquidity_scores) for symbol in liquidity_scores}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest